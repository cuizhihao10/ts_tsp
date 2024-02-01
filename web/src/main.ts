import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import ElementPlus from 'element-plus';
import 'element-plus/dist/index.css';
import locale from 'element-plus/es/locale/lang/zh-cn'
import './theme.css'
import 'uno.css'
import router from './router'
import 'virtual:svg-icons-register'
import SvgIcon from '@/components/SvgIcon/SvgIcon.vue'
import gloalComponent from '@/components/index'
import "@/styles/index.scss"
import cookies from 'vue3-cookies'
// import axios from "axios";
// import VueAxios from "vue-axios";



const app = createApp(App);
const pinia = createPinia();
// app.use(VueAxios, axios);
app.use(cookies);
app.use(pinia);
app.use(router);
app.component('SvgIcon', SvgIcon);
app.use(ElementPlus, 
    { size: 'small', zIndex: 3000, locale 
});
app.use(gloalComponent);
app.mount('#app');

// axios.defaults.baseURL = '/api'  // api 即上面 vue.config.js 中配置的地址
// axios.defaults.headers.post['Content-Type'] = 'application/x-www-form-urlencoded';
// axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';
// app.config.globalProperties.$axios = axios;
