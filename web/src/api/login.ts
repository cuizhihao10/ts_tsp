import request from "@/utils/request";


export const getcode = (email: string) => {
    return request({
        url: "/auth/captcha/email/",
        params: {
            email: email,
        },
        method: "get",
    });
};

export const register = (data: any) => {
    return request({
        url: "/auth/register/",
        method: "post",
        data,
    });
};

export const login = (data: any) => {
    return request({
        url: "/auth/login/",
        method: "post",
        data,
    });
};

export const logout = () => {
    return request({
        url: "/auth/logout/",
        method: "post",
    });
};