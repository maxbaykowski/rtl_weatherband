# rtl_weatherband

`rtl_weatherband` connects to `csdr_server` for NOAA Weather Radio IQ data,
demodulates NFM with NumPy, encodes the resulting PCM with `ffmpeg`, and streams
the encoded audio directly to an Icecast mount.

This is the first iteration. Run one process per station/stream.

## Requirements

Python dependencies are declared in `pyproject.toml`.

External system dependencies:

- `ffmpeg`
- an Icecast server with source credentials

## Configuration

Configuration is JSON5. Frequency is specified in MHz and must be between
`162.4` and `162.55`. NFM deemphasis is enabled by default with
`audio.deemphasis_tau` set to `530` microseconds. Set it to `0` to disable it.

See [config.example.json5](config.example.json5).

## Run

```sh
rtl_weatherband config.example.json5
```

The program requests interleaved float IQ from `csdr_server` at 16000 S/s,
performs NFM demodulation and deemphasis with NumPy, then pipes mono signed
16-bit PCM into `ffmpeg` for MP3 or Ogg Vorbis encoding.
