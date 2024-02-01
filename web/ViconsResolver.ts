import { readdirSync } from 'fs'
import { dirname } from 'path'
// @ts-ignore
import { resolveModule } from 'local-pkg'
import type { ComponentResolver } from 'unplugin-vue-components'

/**
 * key: 图标组件名称
 * value: 模块
 */
let iconPkgMap: Map<string, string>

const iconPkgs: Array<string> = [
  '@vicons/fluent',
  '@vicons/ionicons4',
  '@vicons/ionicons5',
  '@vicons/antd',
  '@vicons/fa',
  '@vicons/material',
  '@vicons/tabler',
  '@vicons/carbon',
]

export function ViconsResolver(): ComponentResolver {
  if (!iconPkgMap) {
    try {
      iconPkgMap = new Map();
      iconPkgs.forEach(pkg => {
        let icons = readdirSync(dirname(resolveModule(pkg as string) as string), { withFileTypes: true })
        .filter((item) => !item.isDirectory() && item.name.match(/^[A-Z][A-Za-z0-9]+\.js$/))
        .map((item) => item.name.replace(/\.js$/, ''))
        icons.forEach(icon => iconPkgMap.set(icon, pkg))
      }) 
    } catch (error) {
      console.error(error)
      throw new Error(`[unplugin-vue-components] failed to load vicons, have you installed it?`)
    }
  }

return {
    type: 'component',
    resolve: (name: string) => {
        if (iconPkgMap.has(name)) {
            const pkg = iconPkgMap.get(name);
            if (pkg) {
                return {
                    name: name,
                    from: pkg
                }
            }
        }
    },
}
}