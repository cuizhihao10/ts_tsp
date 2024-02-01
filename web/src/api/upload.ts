import request from "@/api/axios";
import { useTokenStore } from "@/store/modules/mytoken";

const store = useTokenStore();

export const upload = (data: any) => {
    return request({
        url: "/segmentation/upload/",
        method: "post",
        headers: {
            'Content-Type': 'multipart/form-data',
            'Authorization': 'Bearer ' + store.token.access_token,
        },
        data,
    });
};

export const test = (data: any) => {
    return request({
        url: "/segmentation/test/",
        method: "post",
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + store.token.access_token,
        },
        data,
    });
}
