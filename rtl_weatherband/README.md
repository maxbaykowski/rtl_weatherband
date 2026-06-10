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
Encoder bitrate is configured as `icecast.bitrate` in Kbps.

Valid output sample rates for MP3 and Ogg Vorbis are `8000`, `11025`, `16000`,
`22050`, `24000`, `32000`, `44100`, and `48000` Hz. MP3 bitrate is capped by
sample rate: `8`-`64` Kbps at `8000`/`11025` Hz, `8`-`160` Kbps at
`16000`/`22050`/`24000` Hz, and `32`-`320` Kbps at `32000`/`44100`/`48000` Hz.

See [config.example.json5](config.example.json5).

## Run

```sh
rtl_weatherband config.example.json5
```

The program requests interleaved float IQ from `csdr_server` at 16000 S/s,
performs NFM demodulation and deemphasis with NumPy, then pipes mono signed
16-bit PCM into `ffmpeg` for MP3 or Ogg Vorbis encoding. If IQ audio is not
available because the server is idle, disconnected, refusing connections, or
buffer-underrunning, `rtl_weatherband` continuously feeds silence so Icecast
clients do not see gaps in the source stream.
