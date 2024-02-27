//引入全部的全局组件
import SvgIcon from "./SvgIcon/SvgIcon.vue";

//全局对象
const allGloalComponent = { SvgIcon };
export default {
    install(app: { component: (arg0: string, arg1: any) => void; }) {
        Object.keys(allGloalComponent).forEach((key: string) => {
            if (key in allGloalComponent) {
              const component = allGloalComponent[key as keyof typeof allGloalComponent];
              // do something with component
              app.component(key, component);
            }
        });
    }
};