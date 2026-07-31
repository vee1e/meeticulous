use ndarray::{Array, Array1, Array2, Array3, ArrayD, ArrayViewD, IxDyn};
use once_cell::sync::Lazy;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{TensorRef, ValueType};
use regex::Regex;

use std::fs;
use std::path::Path;

pub type DecoderState = (Array3<f32>, Array3<f32>);

const SUBSAMPLING_FACTOR: usize = 8;
const WINDOW_SIZE: f32 = 0.01;
const MAX_TOKENS_PER_STEP: usize = 3;
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4];

static DECODE_SPACE_RE: Lazy<Result<Regex, regex::Error>> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

#[derive(Debug, Clone)]
pub struct TimestampedResult {
    pub text: String,
    pub timestamps: Vec<f32>,
    pub tokens: Vec<String>,
}

#[derive(thiserror::Error, Debug)]
pub enum ParakeetError {
    #[error("ORT error")]
    Ort(#[from] ort::Error),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model input not found: {0}")]
    InputNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Failed to get tensor shape for input: {0}")]
    TensorShape(String),
}

pub struct ParakeetModel {
    encoder: Session,
    decoder_joint: Session,
    preprocessor: Session,
    vocab: Vec<String>,
    blank_idx: i32,
    vocab_size: usize,
    decoder_output_width: usize,
    decoder_enc_input_rank: usize,
    decoder_enc_hidden_axis: usize,
    decoder_enc_hidden_size: Option<usize>,
    enc_scratch: ArrayD<f32>,
    enc_hidden_marker: Option<usize>,
}

impl Drop for ParakeetModel {
    fn drop(&mut self) {
        log::debug!(
            "Dropping ParakeetModel with {} vocab tokens",
            self.vocab.len()
        );
    }
}

fn tensor_type_and_shape<'a>(
    name: &str,
    vt: &'a ValueType,
) -> Result<(&'a [i64], TensorElementType), ParakeetError> {
    match vt {
        ValueType::Tensor { ty, shape, .. } => Ok((shape, *ty)),
        other => Err(ParakeetError::TensorShape(format!(
            "{name} is not a tensor: {other}"
        ))),
    }
}

fn hidden_axis_and_size(shape: &[i64]) -> (usize, Option<usize>) {
    let mut best: Option<(usize, i64)> = None;
    for (i, &d) in shape.iter().enumerate() {
        if d > 1 && best.map_or(true, |(_, b)| d > b) {
            best = Some((i, d));
        }
    }
    match best {
        Some((i, d)) => (i, Some(d as usize)),
        None => (shape.len().saturating_sub(1), None),
    }
}

impl ParakeetModel {
    pub fn new<P: AsRef<Path>>(model_dir: P, quantized: bool) -> Result<Self, ParakeetError> {
        let encoder = Self::init_session(&model_dir, "encoder-model", Some(1), quantized)?;
        let decoder_joint =
            Self::init_session(&model_dir, "decoder_joint-model", Some(1), quantized)?;
        let preprocessor = Self::init_session(&model_dir, "nemo128", Some(1), false)?;

        let (vocab, blank_idx) = Self::load_vocab(&model_dir)?;
        let vocab_size = vocab.len();

        let (dec_enc_shape, dec_enc_dtype) = {
            let input = decoder_joint
                .inputs
                .iter()
                .find(|i| i.name == "encoder_outputs")
                .ok_or_else(|| ParakeetError::InputNotFound("encoder_outputs".to_string()))?;
            tensor_type_and_shape("decoder 'encoder_outputs'", &input.input_type)?
        };
        let decoder_enc_input_rank = dec_enc_shape.len();
        if decoder_enc_input_rank != 3 {
            return Err(ParakeetError::TensorShape(format!(
                "decoder 'encoder_outputs' input has rank {decoder_enc_input_rank} (expected 3, batch×time×hidden): {dec_enc_shape:?}"
            )));
        }
        let (decoder_enc_hidden_axis, decoder_enc_hidden_size) =
            hidden_axis_and_size(dec_enc_shape);
        if decoder_enc_hidden_axis != 2 {
            return Err(ParakeetError::TensorShape(format!(
                "decoder 'encoder_outputs' input shape {dec_enc_shape:?} is not [batch, time, hidden]; hidden must be the last axis"
            )));
        }
        for (i, &d) in dec_enc_shape.iter().enumerate() {
            if i != decoder_enc_hidden_axis && d > 1 {
                return Err(ParakeetError::TensorShape(format!(
                    "decoder 'encoder_outputs' input shape {dec_enc_shape:?} pins dim {i} to {d}; per-step decode requires batch/time = 1"
                )));
            }
        }

        let (enc_out_shape, enc_out_dtype) = {
            let output = encoder
                .outputs
                .iter()
                .find(|o| o.name == "outputs")
                .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?;
            tensor_type_and_shape("encoder 'outputs'", &output.output_type)?
        };
        if enc_out_shape.len() != decoder_enc_input_rank {
            return Err(ParakeetError::TensorShape(format!(
                "encoder output has rank {} but decoder 'encoder_outputs' expects rank {decoder_enc_input_rank}: {enc_out_shape:?} vs {dec_enc_shape:?}",
                enc_out_shape.len()
            )));
        }
        if enc_out_dtype != dec_enc_dtype {
            return Err(ParakeetError::TensorShape(format!(
                "encoder output dtype {enc_out_dtype:?} != decoder 'encoder_outputs' input dtype {dec_enc_dtype:?}"
            )));
        }
        if let Some(hidden) = decoder_enc_hidden_size {
            if !enc_out_shape.contains(&(hidden as i64)) {
                return Err(ParakeetError::TensorShape(format!(
                    "encoder output shape {enc_out_shape:?} has no hidden dim of size {hidden} required by decoder 'encoder_outputs' input {dec_enc_shape:?}"
                )));
            }
        }
        let enc_hidden_marker = enc_out_shape
            .iter()
            .copied()
            .filter(|&d| d > 1)
            .max()
            .map(|d| d as usize);

        let (dec_out_shape, _) = {
            let output = decoder_joint
                .outputs
                .iter()
                .find(|o| o.name == "outputs")
                .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?;
            tensor_type_and_shape("decoder 'outputs'", &output.output_type)?
        };
        let decoder_output_width = dec_out_shape
            .last()
            .copied()
            .filter(|&d| d >= 0)
            .map(|d| d as usize)
            .unwrap_or(0);
        if decoder_output_width > 0 && decoder_output_width < vocab_size {
            return Err(ParakeetError::TensorShape(format!(
                "vocab.txt declares {vocab_size} tokens but the decoder emits only {decoder_output_width} logits per frame; TDT duration split would be wrong (decoder 'outputs' shape {dec_out_shape:?})"
            )));
        }

        let mut enc_dims = vec![1usize; decoder_enc_input_rank];
        enc_dims[decoder_enc_hidden_axis] = decoder_enc_hidden_size.unwrap_or(1).max(1);
        let enc_scratch = ArrayD::zeros(IxDyn(&enc_dims));

        log::info!(
            "Loaded Parakeet vocabulary with {} tokens, blank_idx={}, decoder output width {} (TDT: {})",
            vocab_size,
            blank_idx,
            decoder_output_width,
            decoder_output_width > vocab_size
        );

        Ok(Self {
            encoder,
            decoder_joint,
            preprocessor,
            vocab,
            blank_idx,
            vocab_size,
            decoder_output_width,
            decoder_enc_input_rank,
            decoder_enc_hidden_axis,
            decoder_enc_hidden_size,
            enc_scratch,
            enc_hidden_marker,
        })
    }

    fn init_session<P: AsRef<Path>>(
        model_dir: P,
        model_name: &str,
        intra_threads: Option<usize>,
        try_quantized: bool,
    ) -> Result<Session, ParakeetError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        // Try quantized version first if requested, fallback to regular version
        let model_filename = if try_quantized {
            let quantized_name = format!("{}.int8.onnx", model_name);
            let quantized_path = model_dir.as_ref().join(&quantized_name);
            if quantized_path.exists() {
                log::info!(
                    "Loading quantized Parakeet model from {}...",
                    quantized_name
                );
                quantized_name
            } else {
                let regular_name = format!("{}.onnx", model_name);
                log::info!(
                    "Quantized model not found, loading regular Parakeet model from {}...",
                    regular_name
                );
                regular_name
            }
        } else {
            let regular_name = format!("{}.onnx", model_name);
            log::info!("Loading Parakeet model from {}...", regular_name);
            regular_name
        };

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?;

        if let Some(threads) = intra_threads {
            builder = builder
                .with_intra_threads(threads)?
                .with_inter_threads(threads)?;
        }

        let session = builder.commit_from_file(model_dir.as_ref().join(&model_filename))?;

        for input in &session.inputs {
            log::info!(
                "Parakeet Model '{}' input: name={}, type={:?}",
                model_filename,
                input.name,
                input.input_type
            );
        }

        Ok(session)
    }

    fn load_vocab<P: AsRef<Path>>(model_dir: P) -> Result<(Vec<String>, i32), ParakeetError> {
        let vocab_path = model_dir.as_ref().join("vocab.txt");
        let content = fs::read_to_string(vocab_path)?;

        let mut max_id = 0;
        let mut tokens_with_ids: Vec<(String, usize)> = Vec::new();
        let mut blank_idx: Option<usize> = None;

        for line in content.lines() {
            let parts: Vec<&str> = line.trim_end().split(' ').collect();
            if parts.len() >= 2 {
                let token = parts[0].to_string();
                if let Ok(id) = parts[1].parse::<usize>() {
                    if token == "<blk>" {
                        blank_idx = Some(id);
                    }
                    tokens_with_ids.push((token, id));
                    max_id = max_id.max(id);
                }
            }
        }

        // Create vocab vector with \u2581 replaced with space
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in tokens_with_ids {
            vocab[id] = token.replace('\u{2581}', " ");
        }

        let blank_idx = blank_idx.ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Missing <blk> token in vocabulary",
            ))
        })? as i32;

        Ok((vocab, blank_idx))
    }

    pub fn preprocess(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_lens: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        log::trace!("Running Parakeet preprocessor inference...");
        let inputs = inputs![
            "waveforms" => TensorRef::from_array_view(waveforms.view())?,
            "waveforms_lens" => TensorRef::from_array_view(waveforms_lens.view())?,
        ];
        let outputs = self.preprocessor.run(inputs)?;

        let features = outputs
            .get("features")
            .ok_or_else(|| ParakeetError::OutputNotFound("features".to_string()))?
            .try_extract_array()?;
        let features_lens = outputs
            .get("features_lens")
            .ok_or_else(|| ParakeetError::OutputNotFound("features_lens".to_string()))?
            .try_extract_array()?;

        Ok((features.to_owned(), features_lens.to_owned()))
    }

    pub fn encode(
        &mut self,
        audio_signal: &ArrayViewD<f32>,
        length: &ArrayViewD<i64>,
    ) -> Result<(ArrayD<f32>, ArrayD<i64>), ParakeetError> {
        log::trace!("Running Parakeet encoder inference...");
        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let encoder_output = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?
            .to_owned();
        let encoded_lengths = outputs
            .get("encoded_lengths")
            .ok_or_else(|| ParakeetError::OutputNotFound("encoded_lengths".to_string()))?
            .try_extract_array()?
            .to_owned();
        drop(outputs);

        let encoder_output = self.normalize_enc_layout(encoder_output)?;

        Ok((encoder_output, encoded_lengths))
    }

    fn normalize_enc_layout(
        &self,
        encoder_output: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, ParakeetError> {
        if encoder_output.ndim() != self.decoder_enc_input_rank {
            return Err(ParakeetError::TensorShape(format!(
                "encoder output has rank {} but decoder expects rank {}: shape {:?}",
                encoder_output.ndim(),
                self.decoder_enc_input_rank,
                encoder_output.shape()
            )));
        }
        // FastConformer exports emit either [batch, time, hidden] or [batch,
        // hidden, time]; normalize to [batch, time, hidden] so decode_sequence
        // can slice along the time axis. The hidden size is the one the
        // decoder's 'encoder_outputs' input metadata pins down (falling back to
        // the encoder's own static output dim when that is fully dynamic).
        let Some(hidden) = self.decoder_enc_hidden_size.or(self.enc_hidden_marker) else {
            return Ok(encoder_output);
        };
        let shape = encoder_output.shape();
        if shape[2] == hidden {
            Ok(encoder_output)
        } else if shape[1] == hidden {
            Ok(encoder_output.permuted_axes(IxDyn(&[0, 2, 1])))
        } else {
            Err(ParakeetError::TensorShape(format!(
                "encoder output shape {shape:?} does not put the decoder's hidden size {hidden} on the time or hidden axis"
            )))
        }
    }

    pub fn create_decoder_state(&self) -> Result<DecoderState, ParakeetError> {
        // Get input shapes from decoder model
        let inputs = &self.decoder_joint.inputs;

        let state1_shape = inputs
            .iter()
            .find(|input| input.name == "input_states_1")
            .ok_or_else(|| ParakeetError::InputNotFound("input_states_1".to_string()))?
            .input_type
            .tensor_shape()
            .ok_or_else(|| ParakeetError::TensorShape("input_states_1".to_string()))?;

        let state2_shape = inputs
            .iter()
            .find(|input| input.name == "input_states_2")
            .ok_or_else(|| ParakeetError::InputNotFound("input_states_2".to_string()))?
            .input_type
            .tensor_shape()
            .ok_or_else(|| ParakeetError::TensorShape("input_states_2".to_string()))?;

        let state1 = Self::zero_decoder_state("input_states_1", state1_shape)?;
        let state2 = Self::zero_decoder_state("input_states_2", state2_shape)?;

        Ok((state1, state2))
    }

    fn zero_decoder_state(name: &str, shape: &[i64]) -> Result<Array3<f32>, ParakeetError> {
        if shape.len() != 3 {
            return Err(ParakeetError::TensorShape(format!(
                "{name} has rank {} (expected 3): {shape:?}",
                shape.len()
            )));
        }
        let d0 = shape[0];
        let d2 = shape[2];
        if d0 < 0 || d2 < 0 {
            return Err(ParakeetError::TensorShape(format!(
                "{name} has dynamic non-batch dims {shape:?}; cannot build a zero decoder state"
            )));
        }
        Ok(Array::zeros((d0 as usize, 1, d2 as usize)))
    }

    pub fn decode_step(
        &mut self,
        prev_tokens: &[i32],
        prev_state: &DecoderState,
        encoder_out: &ArrayViewD<f32>, // [hidden] — a single encoding timestep
    ) -> Result<(ArrayD<f32>, DecoderState), ParakeetError> {
        log::trace!("Running Parakeet decoder inference...");

        // Get last token or blank_idx if empty
        let target_token = prev_tokens.last().copied().unwrap_or(self.blank_idx);

        // Build encoder_outputs to exactly match the decoder's expected input
        // layout ([batch=1, time=1, hidden]) derived from session metadata,
        // reusing a scratch buffer instead of allocating per timestep.
        let features = encoder_out.len();
        if let Some(hidden) = self.decoder_enc_hidden_size {
            if features != hidden {
                return Err(ParakeetError::TensorShape(format!(
                    "encoder timestep has {features} features but decoder 'encoder_outputs' expects {hidden}"
                )));
            }
        }
        if self.enc_scratch.len() != features {
            let mut dims = vec![1usize; self.decoder_enc_input_rank];
            dims[self.decoder_enc_hidden_axis] = features;
            self.enc_scratch = ArrayD::zeros(IxDyn(&dims));
        }
        let shape_err = || {
            ParakeetError::Shape(ndarray::ShapeError::from_kind(
                ndarray::ErrorKind::IncompatibleShape,
            ))
        };
        let src = encoder_out.as_slice().ok_or_else(shape_err)?;
        let dst = self.enc_scratch.as_slice_mut().ok_or_else(shape_err)?;
        dst.copy_from_slice(src);

        let targets = Array2::from_shape_vec((1, 1), vec![target_token])?;
        let target_length = Array1::<i64>::from_vec(vec![1]);

        let inputs = inputs![
            "encoder_outputs" => TensorRef::from_array_view(self.enc_scratch.view())?,
            "targets" => TensorRef::from_array_view(targets.view())?,
            "target_length" => TensorRef::from_array_view(target_length.view())?,
            "input_states_1" => TensorRef::from_array_view(prev_state.0.view())?,
            "input_states_2" => TensorRef::from_array_view(prev_state.1.view())?,
        ];

        let outputs = self.decoder_joint.run(inputs)?;

        let logits = outputs
            .get("outputs")
            .ok_or_else(|| ParakeetError::OutputNotFound("outputs".to_string()))?
            .try_extract_array()?;
        log::trace!(
            "Parakeet Logits shape: {:?}, vocab_size: {}",
            logits.shape(),
            self.vocab_size
        );
        let state1 = outputs
            .get("output_states_1")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_1".to_string()))?
            .try_extract_array()?;
        let state2 = outputs
            .get("output_states_2")
            .ok_or_else(|| ParakeetError::OutputNotFound("output_states_2".to_string()))?
            .try_extract_array()?;

        // Squeeze outputs like Python (remove batch dimension)
        let logits = logits.remove_axis(ndarray::Axis(0));

        // Convert ArrayD back to Array3 to match expected return type
        let state1_3d = state1.to_owned().into_dimensionality::<ndarray::Ix3>()?;
        let state2_3d = state2.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        Ok((logits.to_owned(), (state1_3d, state2_3d)))
    }

    pub fn recognize_batch(
        &mut self,
        waveforms: &ArrayViewD<f32>,
        waveforms_len: &ArrayViewD<i64>,
    ) -> Result<Vec<TimestampedResult>, ParakeetError> {
        // Preprocess and encode
        let (features, features_lens) = self.preprocess(waveforms, waveforms_len)?;
        let (encoder_out, encoder_out_lens) =
            self.encode(&features.view(), &features_lens.view())?;

        // Decode for each batch item
        let mut results = Vec::new();
        for (encodings, &encodings_len) in encoder_out.outer_iter().zip(encoder_out_lens.iter()) {
            let (tokens, timestamps) =
                self.decode_sequence(&encodings.view(), encodings_len as usize)?;
            let result = self.decode_tokens(tokens, timestamps);
            results.push(result);
        }

        Ok(results)
    }

    fn decode_sequence(
        &mut self,
        encodings: &ArrayViewD<f32>, // [time_steps, hidden]
        encodings_len: usize,
    ) -> Result<(Vec<i32>, Vec<usize>), ParakeetError> {
        if encodings.ndim() != 2 {
            return Err(ParakeetError::TensorShape(format!(
                "decode_sequence got encodings with rank {} (expected 2, time×hidden): shape {:?}",
                encodings.ndim(),
                encodings.shape()
            )));
        }
        let encodings_len = encodings_len.min(encodings.shape()[0]);

        let mut prev_state = self.create_decoder_state()?;
        let mut tokens = Vec::new();
        let mut timestamps = Vec::new();

        let mut t = 0;
        let mut emitted_tokens = 0;

        while t < encodings_len {
            let encoder_step = encodings.slice(ndarray::s![t, ..]);
            let (probs, new_state) =
                self.decode_step(&tokens, &prev_state, &encoder_step.into_dyn())?;

            if probs.len() < self.vocab_size
                || (self.decoder_output_width > 0 && probs.len() != self.decoder_output_width)
            {
                return Err(ParakeetError::TensorShape(format!(
                    "decoder emitted {} logits but vocab.txt declares {} tokens (decoder output width {} from session metadata); TDT duration split would be wrong",
                    probs.len(),
                    self.vocab_size,
                    self.decoder_output_width
                )));
            }

            // For TDT models, split output into vocab logits and duration logits
            // output[:vocab_size] = vocabulary logits
            // output[vocab_size:] = duration logits
            let vocab_logits_slice = probs.as_slice().ok_or_else(|| {
                ParakeetError::Shape(ndarray::ShapeError::from_kind(
                    ndarray::ErrorKind::IncompatibleShape,
                ))
            })?;

            let is_tdt = probs.len() > self.vocab_size;
            let (vocab_logits, duration_logits) = if is_tdt {
                let (v, d) = vocab_logits_slice.split_at(self.vocab_size);
                (v, Some(d))
            } else {
                (vocab_logits_slice, None)
            };

            // Get argmax token from vocabulary logits only
            let token = vocab_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i32)
                .unwrap_or(self.blank_idx);

            if token != self.blank_idx {
                prev_state = new_state;
                tokens.push(token);
                timestamps.push(t);
                emitted_tokens += 1;
            }

            if let Some(duration_logits) = duration_logits {
                // TDT: advance by the model's predicted duration (frames to skip).
                let dur_idx = duration_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let mut skip = TDT_DURATIONS.get(dur_idx).copied().unwrap_or(1);

                // Ensure forward progress on blank-with-zero-duration, and cap
                // same-frame emissions to avoid runaway repetition.
                if skip == 0 && (token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP) {
                    skip = 1;
                }
                if skip > 0 {
                    t += skip;
                    emitted_tokens = 0;
                }
            } else {
                // RNN-T greedy: advance one frame on blank or after emission cap.
                if token == self.blank_idx || emitted_tokens >= MAX_TOKENS_PER_STEP {
                    t += 1;
                    emitted_tokens = 0;
                }
            }
        }

        // NEW: Log if no tokens were decoded (helps debugging empty transcriptions)
        if tokens.is_empty() {
            log::debug!(
                "Parakeet decoded zero tokens (all blank) for audio with {} encoding timesteps - audio may be too short or low energy",
                encodings_len
            );
        }

        Ok((tokens, timestamps))
    }

    fn decode_tokens(&self, ids: Vec<i32>, timestamps: Vec<usize>) -> TimestampedResult {
        let tokens: Vec<String> = ids
            .iter()
            .filter_map(|&id| {
                let idx = id as usize;
                if idx < self.vocab.len() {
                    Some(self.vocab[idx].clone())
                } else {
                    None
                }
            })
            .collect();

        let text = match &*DECODE_SPACE_RE {
            Ok(regex) => regex
                .replace_all(&tokens.join(""), |caps: &regex::Captures| {
                    if caps.get(1).is_some() {
                        " "
                    } else {
                        ""
                    }
                })
                .to_string(),
            Err(_) => tokens.join(""), // Fallback if regex failed to compile
        };

        let float_timestamps: Vec<f32> = timestamps
            .iter()
            .map(|&t| WINDOW_SIZE * SUBSAMPLING_FACTOR as f32 * t as f32)
            .collect();

        TimestampedResult {
            text,
            timestamps: float_timestamps,
            tokens,
        }
    }

    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
    ) -> Result<TimestampedResult, ParakeetError> {
        let batch_size = 1;
        let samples_len = samples.len();

        // Create waveforms array [batch_size, samples_len]
        let waveforms = Array2::from_shape_vec((batch_size, samples_len), samples)?.into_dyn();

        // Create waveforms_lens array [batch_size] with the actual length
        let waveforms_lens = Array1::from_vec(vec![samples_len as i64]).into_dyn();

        // Run recognition to get detailed results
        let results = self.recognize_batch(&waveforms.view(), &waveforms_lens.view())?;

        // Extract the first (and only) result
        let timestamped_result = results.into_iter().next().ok_or_else(|| {
            ParakeetError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No transcription result returned",
            ))
        })?;

        Ok(timestamped_result)
    }
}
