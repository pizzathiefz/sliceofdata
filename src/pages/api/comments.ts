import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { EmailMessage } from 'cloudflare:email';
import { createMimeMessage } from 'mimetext';
import { hashIp, guestName } from '../../lib/guest-name';

export const prerender = false;

const RATE_LIMIT_SECONDS = 20;
const NAME_MAX = 50;
const BODY_MAX = 2000;
const NOTIFY_FROM = 'comments@sliceofdata.app';
const NOTIFY_TO = 'pizzathief0@gmail.com';

async function notifyNewComment(path: string, name: string, body: string) {
  try {
    const msg = createMimeMessage();
    msg.setSender({ name: 'sliceofdata', addr: NOTIFY_FROM });
    msg.setRecipient(NOTIFY_TO);
    msg.setSubject(`New comment on ${path}`);
    msg.addMessage({
      contentType: 'text/plain',
      data: `${name} commented on https://sliceofdata.app${path}\n\n${body}`,
    });

    const message = new EmailMessage(NOTIFY_FROM, NOTIFY_TO, msg.asRaw());
    await env.SEND_EMAIL.send(message);
  } catch {
    // best-effort notification — never fail the comment POST because of it
  }
}

export const GET: APIRoute = async ({ url }) => {
  const path = url.searchParams.get('path');
  if (!path) {
    return new Response(JSON.stringify({ error: 'path is required' }), { status: 400 });
  }

  const db = env.DB;
  const { results } = await db
    .prepare(
      'SELECT id, parent_id, name, body, created_at FROM comments WHERE path = ? ORDER BY created_at ASC',
    )
    .bind(path)
    .all();

  return new Response(JSON.stringify({ comments: results }), {
    headers: { 'content-type': 'application/json' },
  });
};

export const POST: APIRoute = async ({ request }) => {
  const db = env.DB;

  let payload: Record<string, unknown>;
  try {
    payload = await request.json();
  } catch {
    return new Response(JSON.stringify({ error: 'invalid JSON' }), { status: 400 });
  }

  const path = typeof payload.path === 'string' ? payload.path : '';
  const name = typeof payload.name === 'string' ? payload.name.trim() : '';
  const body = typeof payload.body === 'string' ? payload.body.trim() : '';
  const honeypot = typeof payload.website === 'string' ? payload.website.trim() : '';
  const parentId = typeof payload.parentId === 'number' ? payload.parentId : null;

  if (!path.startsWith('/') || path.length > 200) {
    return new Response(JSON.stringify({ error: 'invalid path' }), { status: 400 });
  }
  if (name.length > NAME_MAX || !body || body.length > BODY_MAX) {
    return new Response(JSON.stringify({ error: 'invalid name or body' }), { status: 400 });
  }
  if (parentId != null && !Number.isInteger(parentId)) {
    return new Response(JSON.stringify({ error: 'invalid parentId' }), { status: 400 });
  }

  // Honeypot: bots fill hidden fields. Pretend success without writing anything.
  if (honeypot) {
    return new Response(JSON.stringify({ ok: true }), { status: 201 });
  }

  if (parentId != null) {
    const parent = await db
      .prepare('SELECT id FROM comments WHERE id = ? AND path = ? AND parent_id IS NULL')
      .bind(parentId, path)
      .first();
    if (!parent) {
      return new Response(JSON.stringify({ error: 'invalid parentId' }), { status: 400 });
    }
  }

  const ip = request.headers.get('cf-connecting-ip') ?? 'unknown';
  const ipHash = await hashIp(ip);
  const finalName = name || guestName(ipHash);

  const recent = await db
    .prepare(
      "SELECT id FROM comments WHERE ip_hash = ? AND created_at > datetime('now', ?) LIMIT 1",
    )
    .bind(ipHash, `-${RATE_LIMIT_SECONDS} seconds`)
    .first();

  if (recent) {
    return new Response(JSON.stringify({ error: 'too many comments, please wait a moment' }), {
      status: 429,
    });
  }

  const inserted = await db
    .prepare(
      'INSERT INTO comments (path, parent_id, name, body, ip_hash) VALUES (?, ?, ?, ?, ?) RETURNING id, parent_id, name, body, created_at',
    )
    .bind(path, parentId, finalName, body, ipHash)
    .first();

  await notifyNewComment(path, finalName, body);

  return new Response(JSON.stringify({ comment: inserted }), {
    status: 201,
    headers: { 'content-type': 'application/json' },
  });
};
