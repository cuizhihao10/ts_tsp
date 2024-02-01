import { defineConfig } from 'vite'
import path from 'path'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import Icons from 'unplugin-icons/vite'
import IconsResolver from 'unplugin-icons/resolver'
import Components from 'unplugin-vue-components/vite'
import { NaiveUiResolver } from 'unplugin-vue-components/resolvers'
import ViconsResolver from 'unplugin-icons/resolver'
import UnoCSS from 'unocss/vite'
import presetUno from '@unocss/preset-uno'
import presetAttributify from '@unocss/preset-attributify'
import { createSvgIconsPlugin } from 'vite-plugin-svg-icons'
import AutoImport from 'unplugin-auto-import/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

/** 路径查找 */
const pathResolve = (dir: string): string => {
  return resolve(__dirname, '.', dir)
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    UnoCSS({
      presets: [presetUno(), presetAttributify()]
    }),
    Components({
      // dts: true, // ts 环境下要启用  
      resolvers: [NaiveUiResolver(), ElementPlusResolver(), IconsResolver({
        // ep 是 Element Plus 的缩写
        enabledCollections: ['ep'],
      }), ViconsResolver()], // 按需引入 naive-ui 组件
      dirs: ['src/components', 'src/layouts'], // 指定自定义组件存放的文件夹
      dts: path.resolve(pathResolve('components.d.ts')),
    }),
    createSvgIconsPlugin({
      iconDirs: [pathResolve('src/assets/icons')],
      symbolId: 'icon-[dir]-[name]',
    }),
    AutoImport({
      imports: ['vue'],
      resolvers: [ElementPlusResolver(), IconsResolver()],
      dts: path.resolve(pathResolve('auto-imports.d.ts')),
    }),
    Icons({
      autoInstall: true,
    }),
  ],
  resolve: {
    alias: {
      '@': pathResolve('src')
    },
  },
  server: {
    hmr: true,
    // proxy: {
    //   '/api': {
    //     target: 'http://localhost:5000/api',
    //     changeOrigin: true,
    //     rewrite: (path) => path.replace(/^\/api/, '')
    //   }
    // } 
  }
})
