#!/usr/bin/env node
/**
 * 価格帯別出来高の集計が、画面の確かめ（src/lib/volumeProfile.js の parseVolumeProfile）を通るかを見る。
 *
 * Update Volume Profiles が Release に上げる前に、Deploy to GitHub Pages が画面に載せる前に使う。通らない集計を
 * 載せると、画面が「形式を確認してください」と出して何も表示できないため、その前に止める。出すのは件数だけ。
 *
 *   node scripts/check_volume_profile.mjs volume-profile.json
 */
import fs from 'node:fs';
import { PROFILE_WINDOWS, SOURCES, parseVolumeProfile } from '../src/lib/volumeProfile.js';

const file = process.argv[2];
if (!file) {
  console.error('使い方: node scripts/check_volume_profile.mjs volume-profile.json');
  process.exit(2);
}
try {
  const data = parseVolumeProfile(fs.readFileSync(file, 'utf8'));
  const ok = PROFILE_WINDOWS.map((days) => `${days}営業日 ${SOURCES.map((source) =>
    `${source} ${data.stocks.filter((stock) => stock.windows[days][source].status === 'ok').length}`).join('・')}`);
  console.log(`画面の確かめ: ${data.stocks.length}銘柄を読める（表示できる銘柄 ${ok.join(' / ')}）`);
} catch (error) {
  console.error(`::error title=価格帯別出来高::画面の確かめに失敗（${error.message}）。上げない・載せない`);
  process.exit(1);
}
