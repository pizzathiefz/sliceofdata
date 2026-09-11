import type { APIRoute } from 'astro';
import { hashIp, guestName } from '../../lib/guest-name';

export const prerender = false;

export const GET: APIRoute = async ({ request }) => {
  const ip = request.headers.get('cf-connecting-ip') ?? 'unknown';
  const ipHash = await hashIp(ip);

  return new Response(JSON.stringify({ name: guestName(ipHash) }), {
    headers: { 'content-type': 'application/json' },
  });
};
