CREATE TABLE comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  parent_id INTEGER REFERENCES comments (id),
  name TEXT NOT NULL,
  body TEXT NOT NULL,
  ip_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_comments_path ON comments (path, created_at);
CREATE INDEX idx_comments_ip_hash ON comments (ip_hash, created_at);
