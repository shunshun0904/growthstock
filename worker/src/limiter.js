/** One SQLite-backed Durable Object serializes quotas across Worker isolates. */
export function integerSetting(env, name, fallback, max) {
  const value = env[name] ?? String(fallback);
  if (!/^\d+$/.test(String(value)) || Number(value) < 1 || Number(value) > max)
    throw new Error('Invalid quota configuration');
  return Number(value);
}
export function quotaLimits(env) {
  return {
    minute: integerSetting(env, 'CHAT_PER_MINUTE', 3, 5),
    day: integerSetting(env, 'CHAT_PER_DAY', 20, 50),
    month: integerSetting(env, 'CHAT_PER_MONTH', 300, 1000),
  };
}
export function quotaStep(previous, input, now, limits, leaseMs = 35000) {
  const iso = new Date(now).toISOString(), day = iso.slice(0, 10), month = iso.slice(0, 7);
  const minute = Math.floor(now / 60000);
  const s = { ...previous, ips: { ...(previous?.ips || {}) } };
  if (s.month !== month) { s.month = month; s.monthCalls = 0; }
  if (s.day !== day) { s.day = day; s.dayCalls = 0; s.dayAttempts = 0; s.ips = {}; }
  if (s.minute !== minute) { s.minute = minute; s.minuteCalls = 0; s.minuteAttempts = 0; }
  if (s.lease?.until <= now) s.lease = null;
  const usage = () => ({ day, month, callsToday: s.dayCalls, callsThisMonth: s.monthCalls,
    dailyLimit: limits.day, monthlyLimit: limits.month });
  const denied = (code, retryAfter, status = 429) => ({ state: s, result: { allowed: false, code, status, retryAfter, usage: usage() } });
  if (input.action === 'attempt') {
    if (!/^[a-f0-9]{64}$/.test(input.fingerprint || '')) throw new Error('Invalid fingerprint');
    if (s.minuteAttempts >= 60 || s.dayAttempts >= 500) return denied('AUTH_RATE_LIMIT', 60);
    const current = s.ips[input.fingerprint];
    const ip = current ? { ...current } : { minute, minuteAttempts: 0, dayAttempts: 0 };
    if (!current && Object.keys(s.ips).length >= 256) return denied('AUTH_RATE_LIMIT', 60);
    if (ip.minute !== minute) { ip.minute = minute; ip.minuteAttempts = 0; }
    if (ip.minuteAttempts >= 10 || ip.dayAttempts >= 100) return denied('AUTH_RATE_LIMIT', 60);
    ip.minuteAttempts++; ip.dayAttempts++; s.minuteAttempts++; s.dayAttempts++;
    s.ips[input.fingerprint] = ip;
  } else if (input.action === 'reserve') {
    if (s.lease) return denied('BUSY', Math.max(1, Math.ceil((s.lease.until - now) / 1000)), 409);
    if (s.minuteCalls >= limits.minute) return denied('RATE_LIMIT', 60 - Math.floor(now / 1000) % 60);
    if (s.dayCalls >= limits.day) return denied('DAILY_LIMIT', 86400 - Math.floor(now / 1000) % 86400);
    if (s.monthCalls >= limits.month) {
      const next = new Date(Date.UTC(Number(iso.slice(0, 4)), Number(iso.slice(5, 7)), 1)).getTime();
      return denied('MONTHLY_LIMIT', Math.ceil((next - now) / 1000));
    }
    s.minuteCalls++; s.dayCalls++; s.monthCalls++;
    s.lease = { id: input.id, until: now + leaseMs };
  } else if (input.action === 'release') {
    if (s.lease?.id === input.id) s.lease = null;
  } else if (input.action !== 'status') throw new Error('Unknown quota operation');
  return { state: s, result: { allowed: true, usage: usage() } };
}
export class AiLimiter {
  constructor(ctx, env) { this.ctx = ctx; this.env = env; }
  async fetch(request) {
    try {
      const input = await request.json();
      const limits = quotaLimits(this.env);
      const leaseMs = integerSetting(this.env, 'UPSTREAM_TIMEOUT_MS', 25000, 30000) + 5000;
      const result = await this.ctx.storage.transaction(async tx => {
        const step = quotaStep(await tx.get('quota-v1'), input, Date.now(), limits, leaseMs);
        await tx.put('quota-v1', step.state);
        return step.result;
      });
      return Response.json(result);
    } catch {
      return Response.json({ allowed: false, code: 'LIMITER_UNAVAILABLE', status: 503 }, { status: 503 });
    }
  }
}
