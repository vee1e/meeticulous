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
    let abl = inInputData.pointee
    let buf = abl.mBuffers
    let byteCount = Int(buf.mDataByteSize)
    guard byteCount > 0, let data = buf.mData else { return noErr }
    let floatCount = byteCount / MemoryLayout<Float>.size
    let floats = data.assumingMemoryBound(to: Float.self)
    // Write raw f32le mono (tap is mono global)
    let ptr = UnsafeRawPointer(floats).assumingMemoryBound(to: UInt8.self)
    let slice = UnsafeBufferPointer(start: ptr, count: floatCount * MemoryLayout<Float>.size)
    let written = fwrite(slice.baseAddress, 1, slice.count, stdout)
    if written != slice.count {
        gState.running = false
    }
    return noErr
}

func main() {
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

    fputs("READY sample_rate=\(Int(rate)) device=\(outName)\n", stderr)
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
