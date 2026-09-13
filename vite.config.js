import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';
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

// base: './' -> GitHub Pages のサブパス配信でも、ローカルの file/preview でも動くようにする
export default defineConfig({
  plugins: [react()],
  base: './',
  define: {
    __BUILD_ID__: JSON.stringify(buildId()),
    __BUILT_AT__: JSON.stringify(new Date().toISOString()),
  },
  build: { outDir: 'dist', sourcemap: false },
});
