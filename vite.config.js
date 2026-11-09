import { execSync } from 'node:child_process';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

/**
 * ビルド識別子。コミットの短縮 SHA、取れなければ日時。
 *
 * 何に使うか
 * ---------
 * 1. 画面の右上に出す。「更新されていない」と言われたときに、利用者が
 *    どの版を見ているのかを一目で確かめられる。これが無いと、
 *    サーバーが古いのかブラウザが古いのかを切り分けられない。
 * 2. データ JSON の取得 URL に付ける。デプロイのたびに URL が変わるので、
 *    ブラウザにも CDN にも古い JSON を返す余地が無くなる。
 *    データが変わるのは必ずデプロイを伴うので、これで十分。
 */
function buildId() {
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString().trim();
  } catch {
    return new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
  }
}

function protectPublicEnvironment() {
  return { name: 'protect-public-environment', config(_, { mode }) {
    const publicEnv = loadEnv(mode, process.cwd(), 'VITE_');
    const names = Object.keys(publicEnv)
      .filter(name => /^VITE_.*(?:API_?KEY|PASSWORD|SECRET|TOKEN)/i.test(name));
    if (names.length) throw new Error(`秘密情報をVITE_変数に設定できません。Worker Secretsを使用してください（${names.join(', ')}）。`);
    if (publicEnv.VITE_PLAMO_WORKER_URL) {
      try {
        const url = new URL(publicEnv.VITE_PLAMO_WORKER_URL.trim());
        const local = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
        if ((url.protocol !== 'https:' && !(local && url.protocol === 'http:'))
            || url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error();
      } catch { throw new Error('VITE_PLAMO_WORKER_URLには認証情報・クエリを含まないHTTPSのルートURLを指定してください。'); }
    }
  } };
}

// base: './' -> GitHub Pages のサブパス配信でも、ローカルの file/preview でも動くようにする
export default defineConfig({
  plugins: [protectPublicEnvironment(), react()],
  base: './',
  define: {
    __BUILD_ID__: JSON.stringify(buildId()),
    __BUILT_AT__: JSON.stringify(new Date().toISOString()),
  },
  build: { outDir: 'dist', sourcemap: false },
});
