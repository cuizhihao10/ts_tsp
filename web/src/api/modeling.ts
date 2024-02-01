import request from "@/api/axios";
// import { useTokenStore } from "@/store/modules/mytoken";

// const store = useTokenStore();

export const modeling = (data: any) => {
    return request({
        url: "/segmentation/modeling/",
        method: "post",
        headers: {
            'Content-Type': 'application/json',
        },
        data,
        timeout: 1000 * 600,
        signal: data.signal,
    });
};

export const classific_modeling = (data: any) => {
    return request({
        url: "/classification/classific_modeling/",
        method: "post",
        headers: {
            'Content-Type': 'application/json',
        },
        data,
        timeout: 1000 * 60,
    });
}