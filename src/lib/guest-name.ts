export async function hashIp(ip: string): Promise<string> {
  const data = new TextEncoder().encode(`sliceofdata-comments:${ip}`);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export function guestName(ipHash: string): string {
  return `Guest ${ipHash.slice(0, 4)}`;
}
