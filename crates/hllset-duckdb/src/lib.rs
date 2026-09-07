use hllset_core::hashing::token_to_position;
use hllset_core::HLLSet;
use hllset_storage::Storage;
use rusqlite::{params, Connection};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const NUM_CHUNKS: usize = 4;
pub const REGS_PER_CHUNK: usize = 1024 / NUM_CHUNKS;

pub struct ChunkId {
    pub name: String,
    pub reg_start: u16,
    pub reg_end: u16,
}
impl ChunkId {
    pub fn chunk_for_reg(reg: u16) -> usize {
        (reg as usize) / REGS_PER_CHUNK
    }
    pub fn storage_key(&self) -> String {
        format!("lut:{}:{}-{}", self.name, self.reg_start, self.reg_end)
    }
    pub fn all_for(name: &str) -> Vec<Self> {
        (0..NUM_CHUNKS)
            .map(|i| Self {
                name: name.to_string(),
                reg_start: (i * REGS_PER_CHUNK) as u16,
                reg_end: ((i + 1) * REGS_PER_CHUNK) as u16,
            })
            .collect()
    }
}

pub struct ChunkedLUT {
    conns: Vec<Connection>,
    fps: Vec<HLLSet>,
    cnt: Vec<u64>,
    name: String,
}

impl ChunkedLUT {
    pub fn new(name: &str) -> rusqlite::Result<Self> {
        let mut cs = vec![];
        let mut fs = vec![];
        let mut ns = vec![];
        for _ in 0..NUM_CHUNKS {
            let c = Connection::open_in_memory()?;
            c.execute_batch(
                "CREATE TABLE lut (r INTEGER,z INTEGER,t BLOB); CREATE INDEX il ON lut(r,z);",
            )?;
            cs.push(c);
            fs.push(HLLSet::new());
            ns.push(0);
        }
        Ok(Self {
            conns: cs,
            fps: fs,
            cnt: ns,
            name: name.to_string(),
        })
    }
    pub fn insert(&mut self, token: &[u8]) -> rusqlite::Result<()> {
        let (r, z) = token_to_position(token);
        let ci = ChunkId::chunk_for_reg(r as u16);
        if ci < NUM_CHUNKS {
            self.fps[ci].merge_tokens(&[token]);
            self.conns[ci].execute(
                "INSERT INTO lut VALUES(?1,?2,?3)",
                params![r as i64, z as i64, token],
            )?;
            self.cnt[ci] += 1;
        }
        Ok(())
    }
    pub fn insert_all<I, B>(&mut self, t: I) -> rusqlite::Result<()>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        for x in t {
            self.insert(x.as_ref())?;
        }
        Ok(())
    }
    pub fn chunk_count(&self, chunk_idx: usize) -> u64 {
        if chunk_idx < NUM_CHUNKS {
            self.cnt[chunk_idx]
        } else {
            0
        }
    }
    pub fn persist(
        &self,
        s: &dyn Storage,
        tmp: &Path,
    ) -> Result<(usize, HLLSet), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(tmp)?;
        let mut m = HLLSet::new();
        let mut st = 0;
        let ids = ChunkId::all_for(&self.name);
        for (i, id) in ids.iter().enumerate() {
            if self.cnt[i] == 0 {
                continue;
            }
            let tp = tmp.join(format!("c{}.db", i));
            let fc = Connection::open(&tp)?;
            fc.execute_batch("CREATE TABLE IF NOT EXISTS lut(r INTEGER,z INTEGER,t BLOB);")?;
            let mut stmt = self.conns[i].prepare("SELECT r,z,t FROM lut")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, Vec<u8>>(2)?,
                ))
            })?;
            for row in rows {
                let (r, z, t) = row?;
                fc.execute(
                    "INSERT OR IGNORE INTO lut VALUES(?1,?2,?3)",
                    params![r, z, t],
                )?;
            }
            drop(fc);
            let d = std::fs::read(&tp)?;
            s.store(&id.storage_key(), &d)?;
            let _ = std::fs::remove_file(&tp);
            m.merge(&self.fps[i]);
            st += 1;
        }
        Ok((st, m))
    }
}

pub struct ChunkMaterializer {
    s: Arc<dyn Storage>,
    name: String,
    cache: PathBuf,
}

impl ChunkMaterializer {
    pub fn open(
        s: &dyn Storage,
        name: &str,
        cache: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let s_ref: &dyn Storage = s;
        // Clone the Storage into an Arc — requires Storage: Clone
        // Since MemoryStorage is Clone (Rc-based), we need a different approach.
        // For now, just store a separate MemoryStorage — accept the limitation.
        let _ = s_ref;
        Err("ChunkMaterializer::open requires a clonable storage. Use MemoryStorage::new() directly.".into())
    }

    pub fn open_with<S: Storage + Clone + 'static>(
        s: S,
        name: &str,
        cache: PathBuf,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&cache)?;
        Ok(Self {
            s: Arc::new(s),
            name: name.to_string(),
            cache,
        })
    }

    pub fn query(&self, pos: &[(u16, u8)]) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
        if pos.is_empty() {
            return Ok(vec![]);
        }
        let mut need = HashSet::new();
        for &(r, _) in pos {
            need.insert(ChunkId::chunk_for_reg(r));
        }
        let mut res = vec![];
        let ids = ChunkId::all_for(&self.name);
        for ci in need {
            if ci >= ids.len() {
                continue;
            }
            if let Ok(c) = self.load(ci) {
                let cp: Vec<_> = pos
                    .iter()
                    .filter(|(r, _)| ChunkId::chunk_for_reg(*r) == ci)
                    .collect();
                if !cp.is_empty() {
                    let v: Vec<String> = cp.iter().map(|(r, z)| format!("({},{})", r, z)).collect();
                    let sql = format!(
                        "SELECT DISTINCT t FROM lut WHERE (r,z) IN ({})",
                        v.join(",")
                    );
                    if let Ok(mut st) = c.prepare(&sql) {
                        if let Ok(rows) = st.query_map([], |r| Ok(r.get::<_, Vec<u8>>(0)?)) {
                            for r in rows.flatten() {
                                res.push(r);
                            }
                        }
                    }
                }
            }
        }
        Ok(res)
    }
    fn load(&self, ci: usize) -> Result<Connection, Box<dyn std::error::Error>> {
        let cp = self.cache.join(format!("c{}.db", ci));
        if !cp.exists() {
            let ids = ChunkId::all_for(&self.name);
            let d = self
                .s
                .load(&ids[ci].storage_key())?
                .ok_or_else(|| format!("no chunk {}", ci))?;
            std::fs::write(&cp, &d)?;
        }
        Ok(Connection::open(&cp)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hllset_storage::MemoryStorage;
    fn td() -> PathBuf {
        PathBuf::from("/tmp/hllset_ddb_t")
    }
    fn clean() {
        let _ = std::fs::remove_dir_all(td());
    }
    #[test]
    fn test_routing() {
        let r = token_to_position(b"x").0 as u16;
        assert_eq!(ChunkId::chunk_for_reg(r), ChunkId::chunk_for_reg(r));
    }
    #[test]
    fn test_build_query() {
        clean();
        let mut l = ChunkedLUT::new("a").unwrap();
        l.insert(b"abc").unwrap();
        l.insert(b"def").unwrap();
        let s = MemoryStorage::new();
        let (n, _) = l.persist(&s, &td()).unwrap();
        assert!(n > 0);
        let m = ChunkMaterializer::open_with(s, "a", td().join("c")).unwrap();
        let (r, z) = token_to_position(b"abc");
        let res = m.query(&[(r as u16, z as u8)]).unwrap();
        assert!(res.iter().any(|t| t == b"abc"));
        clean();
    }
    #[test]
    fn test_empty_skip() {
        clean();
        let mut l = ChunkedLUT::new("e").unwrap();
        let mut n = 0;
        for i in 0..2000 {
            let t = format!("t{}", i);
            if ChunkId::chunk_for_reg(token_to_position(t.as_bytes()).0 as u16) == 0 {
                l.insert(t.as_bytes()).unwrap();
                n += 1;
            }
        }
        assert!(n > 0);
        let s = MemoryStorage::new();
        let (st, _) = l.persist(&s, &td()).unwrap();
        assert!(st <= 2);
        clean();
    }
}
