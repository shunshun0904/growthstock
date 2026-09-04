import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base: './' -> GitHub Pages のサブパス配信でも、ローカルの file/preview でも動くようにする
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist', sourcemap: false },
});
