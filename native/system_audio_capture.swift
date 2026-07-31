// meeticulous-system-audio: macOS system-audio capture via Core Audio process tap.
// Writes little-endian f32 mono PCM to stdout after a text header line on stderr:
//   READY sample_rate=<hz> device=<name>
//
// Requires Audio Capture permission (macOS 14.4+) for the host process
// (Terminal / iTerm when launched from a shell).

import Foundation
import AVFoundation
import CoreAudio
import AudioToolbox

private final class CaptureState {
    var sampleRate: Double = 48_000
    var running = true
}

private var gState = CaptureState()

private func die(_ msg: String, code: Int32 = 1) -> Never {
    fputs("ERROR \(msg)\n", stderr)
    fflush(stderr)
    exit(code)
}

private func defaultOutputDeviceID() -> AudioDeviceID {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    let err = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        0,
        nil,
        &size,
        &deviceID
    )
    if err != noErr {
        die("kAudioHardwarePropertyDefaultOutputDevice failed: \(err)")
    }
    return deviceID
}

private func deviceUID(_ deviceID: AudioDeviceID) -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    var err = AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size)
    if err != noErr { return "unknown" }
    var cfUID: Unmanaged<CFString>?
    err = withUnsafeMutablePointer(to: &cfUID) { ptr in
        AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, ptr)
    }
    if err != noErr { return "unknown" }
    return cfUID?.takeRetainedValue() as String? ?? "unknown"
}

private func deviceName(_ deviceID: AudioDeviceID) -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    var err = AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size)
    if err != noErr { return "Default Output" }
    var cfName: Unmanaged<CFString>?
    err = withUnsafeMutablePointer(to: &cfName) { ptr in
        AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, ptr)
    }
    if err != noErr { return "Default Output" }
    return cfName?.takeRetainedValue() as String? ?? "Default Output"
}

private func ioProc(
    _ inDevice: AudioObjectID,
    _ inNow: UnsafePointer<AudioTimeStamp>,
    _ inInputData: UnsafePointer<AudioBufferList>,
    _ inInputTime: UnsafePointer<AudioTimeStamp>,
    _ outOutputData: UnsafeMutablePointer<AudioBufferList>,
    _ inOutputTime: UnsafePointer<AudioTimeStamp>,
    _ inClientData: UnsafeMutableRawPointer?
) -> OSStatus {
    // The tap may deliver interleaved or multi-channel data depending on the
    // aggregate format. Downmix everything to mono Float32; bail loudly if a
    // buffer cannot be interpreted as Float32.
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
    var samples: [Float] = []
    samples.reserveCapacity(4096)
    for buf in buffers {
        let byteCount = Int(buf.mDataByteSize)
        guard byteCount > 0, let data = buf.mData else { continue }
        guard byteCount % MemoryLayout<Float>.size == 0 else {
            fputs("ERROR ioProc: tap delivered a non-Float32 buffer (\(byteCount) bytes)\n", stderr)
            fflush(stderr)
            exit(1)
        }
        let ch = max(1, min(Int(buf.mNumberChannels), 64))
        let floatCount = byteCount / MemoryLayout<Float>.size
        let floats = data.assumingMemoryBound(to: Float.self)
        if ch == 1 {
            samples.append(contentsOf: UnsafeBufferPointer(start: floats, count: floatCount))
        } else {
            // Interleaved multi-channel → average channels into mono.
            let frames = floatCount / ch
            var mono = [Float](repeating: 0, count: frames)
            for f in 0..<frames {
                var sum: Float = 0
                for c in 0..<ch {
                    sum += floats[f * ch + c]
                }
                mono[f] = sum / Float(ch)
            }
            samples.append(contentsOf: mono)
        }
    }
    guard !samples.isEmpty else { return noErr }
    let written = samples.withUnsafeBytes { raw -> Int in
        fwrite(raw.baseAddress!, 1, raw.count, stdout)
    }
    if written != samples.count * MemoryLayout<Float>.size {
        gState.running = false
    }
    return noErr
}

func main() {
    // Unbuffered stdout: PCM + READY must stream immediately.
    setvbuf(stdout, nil, _IONBF, 0)
    // Default output (what the user hears — Zoom/Meet/etc.)
    let outID = defaultOutputDeviceID()
    let outUID = deviceUID(outID)
    let outName = deviceName(outID)

    // Global mono process tap excluding nothing → all system audio
    let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    desc.name = "meeticulous-system-tap"
    desc.isPrivate = true

    var tapID = AudioObjectID(kAudioObjectUnknown)
    var err = AudioHardwareCreateProcessTap(desc, &tapID)
    if err != noErr {
        die(
            "AudioHardwareCreateProcessTap failed (\(err)). " +
            "Grant Audio Capture permission: System Settings → Privacy & Security → Audio Capture " +
            "(enable Terminal/iTerm), then fully quit and reopen the terminal."
        )
    }

    // Aggregate device with ONLY the tap (Meetily pattern — avoids echo)
    let tapUUID = desc.uuid.uuidString
    let aggUID = UUID().uuidString
    let aggDesc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: "meeticulous-audio-tap",
        kAudioAggregateDeviceUIDKey as String: aggUID,
        kAudioAggregateDeviceIsPrivateKey as String: true,
        kAudioAggregateDeviceIsStackedKey as String: false,
        kAudioAggregateDeviceTapAutoStartKey as String: true,
        kAudioAggregateDeviceMainSubDeviceKey as String: outUID,
        kAudioAggregateDeviceTapListKey as String: [
            [kAudioSubTapUIDKey as String: tapUUID]
        ],
    ]

    var aggID = AudioObjectID(kAudioObjectUnknown)
    err = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &aggID)
    if err != noErr {
        _ = AudioHardwareDestroyProcessTap(tapID)
        die("AudioHardwareCreateAggregateDevice failed: \(err)")
    }

    // Sample rate from aggregate / default output
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var rate = 48_000.0
    var size = UInt32(MemoryLayout<Double>.size)
    _ = AudioObjectGetPropertyData(aggID, &addr, 0, nil, &size, &rate)
    gState.sampleRate = rate

    var procID: AudioDeviceIOProcID?
    err = AudioDeviceCreateIOProcID(aggID, ioProc, nil, &procID)
    if err != noErr || procID == nil {
        _ = AudioHardwareDestroyAggregateDevice(aggID)
        _ = AudioHardwareDestroyProcessTap(tapID)
        die("AudioDeviceCreateIOProcID failed: \(err)")
    }

    err = AudioDeviceStart(aggID, procID)
    if err != noErr {
        _ = AudioDeviceDestroyIOProcID(aggID, procID!)
        _ = AudioHardwareDestroyAggregateDevice(aggID)
        _ = AudioHardwareDestroyProcessTap(tapID)
        die("AudioDeviceStart failed: \(err) — check Audio Capture permission")
    }

    // Report the ACTUAL stream format the tap delivers (may differ from the
    // nominal device rate). Fall back to the nominal rate on query failure.
    var fmtAddr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamFormat,
        mScope: kAudioDevicePropertyScopeInput,
        mElement: kAudioObjectPropertyElementMain
    )
    var fmtSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    var streamFormat = AudioStreamBasicDescription()
    var actualRate = rate
    if AudioObjectGetPropertyData(aggID, &fmtAddr, 0, nil, &fmtSize, &streamFormat) == noErr,
       streamFormat.mSampleRate > 0 {
        actualRate = streamFormat.mSampleRate
    }
    gState.sampleRate = actualRate

    fputs("READY sample_rate=\(Int(actualRate)) device=\(outName)\n", stderr)
    fflush(stderr)

    // Run until parent closes stdin or SIGTERM
    signal(SIGTERM) { _ in gState.running = false }
    signal(SIGINT) { _ in gState.running = false }

    while gState.running {
        // Parent dying closes our stdout → fwrite fails → running=false
        Thread.sleep(forTimeInterval: 0.05)
        if ferror(stdout) != 0 {
            gState.running = false
        }
    }

    _ = AudioDeviceStop(aggID, procID)
    _ = AudioDeviceDestroyIOProcID(aggID, procID!)
    _ = AudioHardwareDestroyAggregateDevice(aggID)
    _ = AudioHardwareDestroyProcessTap(tapID)
}

main()
