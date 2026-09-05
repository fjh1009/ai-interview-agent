import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 3001,
    proxy: {
      '/api': {
        // 后端地址（Docker 部署在本地 8006 端口）
        target: 'http://localhost:8006',
        changeOrigin: true
      }
    }
  }
})
