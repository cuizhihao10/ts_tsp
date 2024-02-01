import axios, { type AxiosRequestHeaders } from 'axios'
import { useTokenStore } from '@/store/modules/mytoken'
import { refresh_token } from '@/api/user'
import router from '@/router/index'

const instance = axios.create({
    baseURL: 'http://127.0.0.1:5000',
    timeout: 5000,
    headers: {'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'}, 
})

const refinstance = axios.create({
    baseURL: 'http://127.0.0.1:5000',
    timeout: 5000,
    headers: {'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'},
    withCredentials: true,
})

// 请求拦截器
instance.interceptors.request.use( config => {
    if (!config.headers) {
        config.headers = {} as AxiosRequestHeaders;
    }

    const store = useTokenStore();
    config.headers.Authorization = 'Bearer ' + store.token.access_token;

    // 在发送请求之前做些什么
    return config;
}, error => {
    // 对请求错误做些什么
    return Promise.reject(error);
});

// 响应拦截器
instance.interceptors.response.use( 
    (response) => response,
    async (error) => {
        // console.log(error.response)
        // 对响应错误做点什么
        if (error.response.status === 401) {
            const store = useTokenStore();
            store.clearToken();
            const { data } = await refresh_token();
            if (data.code === 200) {
                store.refreah_access_token(data.data.access_token);
                return instance(error.config);
            } else {
                ElMessage.error('登录已过期，请重新登录');
                router.push({ name: 'login', query: { redirect: router.currentRoute.value.fullPath }});
                return 
            }
        // } else if (error.response.status === 500){
        //     console.log(error.response.data.message, 14523524355555555555555555555555);
        //     ElMessage.error('服务器错误，请稍后再试');
        //     return Promise.reject(error);
        }
    return Promise.reject(error);
});

export default instance;
export const refrequest = refinstance;