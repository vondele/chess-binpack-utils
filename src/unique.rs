use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek};
use std::path::Path;

use sfbinpack::CompressedTrainingDataEntryReader;
use shakmaty::{
    CastlingMode, Chess, EnPassantMode, Position, fen::Fen, uci::UciMove, zobrist::Zobrist64,
    zobrist::ZobristHash,
};
use viriformat::dataformat::Game as ViriGame;

use crate::cli::Backend;
use crate::error::{Error, Result};

// https://oertl.github.io/hyperloglog-sketch-estimation-paper/paper/paper.pdf
const BITSUSED: u32 = 20;
struct HyperLogLog{
    m: [u8; 1 << BITSUSED],
}
impl HyperLogLog{
    fn new() -> HyperLogLog {
        HyperLogLog{
            m: [0; 1 << BITSUSED],
        }
    }
    fn tau(mut x: f64) -> f64{
        if x == 0.0 || x == 1.0 {
            return 0.0;
        }
        let mut y: f64 = 1.0;
        let mut z: f64 = 1.0 - x;
        let mut zp: f64 = 2.0;
        while zp != z{
            x = x.sqrt();
            zp = z;
            y /= 2.0;
            z -= (1.0 - x).powf(2.0) * y;
        }
        z/3.0
    }
    fn sigma(mut x: f64) -> f64 {
        if x == 1.0 {
            return f64::INFINITY;
        }
        let mut y: f64 = 1.0;
        let mut z: f64 = x;
        let mut zp: f64 = 2.0;
        while zp != z {
            x *= x;
            zp = z;
            z += x * y;
            y *= 2.0;
        }
        return z;
    }
    fn add(&mut self, data: u64) {
        let j: usize = (data >> (64-BITSUSED)) as usize;
        let w: u64 = data << BITSUSED;
        self.m[j] = (self.m[j]).max(w.trailing_zeros() as u8);
    }
    fn count(&self) -> u64 {
        let size: u64 = (1 as u64) << BITSUSED;
        let mut c: [u32; (64-BITSUSED+2) as usize] = [0; (64-BITSUSED+2) as usize];
        for x in self.m {
            let k:i32 = (x as i32 - BITSUSED as i32 + 1).max(0);
            c[k as usize] += 1;
        }
        let mut z: f64 = size as f64 * HyperLogLog::tau(1.0 - c[(64-BITSUSED+1) as usize] as f64 / size as f64);
        for k in (1..64-BITSUSED+1).rev() {
            z = 0.5 * (z + c[k as usize] as f64);
        }
        z += size as f64 * HyperLogLog::sigma(c[0] as f64 / size as f64);
        return (1.0/(2.0_f64 * (2.0_f64).ln()) * size as f64 * size as f64 / z) as u64;
    }
}

pub fn unique_positions_from_path(
    path: &Path,
    limit: Option<usize>,
    backend: Backend,
) -> Result<u64> {
    let file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    unique_positions_from_file(file, limit, backend)
}

pub fn unique_positions_from_file<T: Read + Seek>(
    file: T,
    limit: Option<usize>,
    backend: Backend,
) -> Result<u64> {
    match backend {
        Backend::Sfbinpack => unique_sf(file, limit),
        Backend::Viriformat => unique_viriformat(file, limit),
    }
}

fn unique_sf<T: Read + Seek>(file: T, limit: Option<usize>) -> Result<u64> {
    let mut reader = CompressedTrainingDataEntryReader::new(file)?;
    let mut position = Chess::default();
    let mut unique: HyperLogLog = HyperLogLog::new();
    let mut new_game = true;
    let mut count = 0usize;

    while reader.has_next() {
        let entry = reader.next();

        if new_game {
            let fen_str = entry.pos.fen().map_err(|error| {
                Error::InvalidGameData(format!("entry FEN could not be read: {error:?}"))
            })?;
            let fen = Fen::from_ascii(fen_str.as_bytes())
                .map_err(|_| Error::InvalidFen(fen_str.clone()))?;
            position = fen.into_position(CastlingMode::Standard).map_err(|_| {
                Error::InvalidGameData(format!("invalid position from FEN: {fen_str}"))
            })?;
            new_game = false;
        }

        let hash = position.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
        unique.add(hash.0);

        if reader.has_next() && reader.is_next_entry_continuation() {
            let uci: UciMove = entry.mv.as_uci().parse().map_err(|error| {
                Error::InvalidGameData(format!("invalid UCI move in sfbinpack stream: {error}"))
            })?;
            let chess_move = uci.to_move(&position).map_err(|error| {
                Error::InvalidGameData(format!("illegal move in sfbinpack stream: {error}"))
            })?;
            position.play_unchecked(&chess_move);
        } else {
            new_game = true;
        }

        count += 1;
        if limit.is_some_and(|limit| count >= limit) {
            break;
        }
    }

    Ok(unique.count() as u64)
}

fn unique_viriformat<T: Read + Seek>(file: T, limit: Option<usize>) -> Result<u64> {
    let mut reader = BufReader::new(file);
    let mut unique: HyperLogLog = HyperLogLog::new();
    let mut processed = 0usize;

    loop {
        match ViriGame::deserialise_from(&mut reader, Vec::new()) {
            Ok(game) => {
                let (board, _, _, _) = game.initial_position.unpack();
                let fen_str = board.to_string();
                let fen = Fen::from_ascii(fen_str.as_bytes())
                    .map_err(|_| Error::InvalidFen(fen_str.clone()))?;
                let mut position: Chess =
                    fen.into_position(CastlingMode::Standard).map_err(|_| {
                        Error::InvalidGameData(format!(
                            "unable to convert FEN to position: {fen_str}"
                        ))
                    })?;

                for (mv, _) in &game.moves {
                    let hash = position.zobrist_hash::<Zobrist64>(EnPassantMode::Legal);
                    unique.add(hash.0);

                    processed += 1;
                    if limit.is_some_and(|limit| processed >= limit) {
                        return Ok(unique.count() as u64);
                    }

                    let uci_string = mv.display(false).to_string();
                    let uci: UciMove = uci_string.parse().map_err(|error| {
                        Error::InvalidGameData(format!(
                            "invalid UCI move in viriformat stream ({uci_string}): {error}"
                        ))
                    })?;
                    let chess_move = uci.to_move(&position).map_err(|error| {
                        Error::InvalidGameData(format!(
                            "illegal move in viriformat stream ({uci_string}): {error}"
                        ))
                    })?;
                    position.play_unchecked(&chess_move);
                }
            }
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(error) => {
                return Err(Error::Io {
                    path: Path::new("<viriformat stream>").to_path_buf(),
                    source: error,
                });
            }
        }
    }

    Ok(unique.count() as u64)
}
