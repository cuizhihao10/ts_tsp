import request, { refrequest } from '@/api/axios'
import { useTokenStore } from '@/store/modules/mytoken'

type UserInfo = {
    username: string
    logined: boolean
    uid: number
}


let promiseRT: Promise<any> | null = null;
let isRefreshing = false;
export const refresh_token = () => {
    if (isRefreshing) {
        return promiseRT;
    }
    isRefreshing = true;
    const store = useTokenStore();
    const refresh_token = store.token.refresh_token;
    promiseRT =  refrequest({  
        url: "/user/refresh/",
        method: "post",
        headers: {
            'Authorization': 'Bearer ' + refresh_token
        },
    }).finally(() => {
        isRefreshing = false;
    })

    return promiseRT;
}

export const getInfo = () => {
    return request<UserInfo>({  
        url: "/user/info/",
        method: "get",
    })        
}

export const logout = () => {
    return request({  
        url: "/user/logout/",
        method: "post",
    })        
}