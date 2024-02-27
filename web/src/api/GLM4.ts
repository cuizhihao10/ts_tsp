import request from '@/api/axios'

export const GLM4_response = (data: object) => {
    return request({
        url: '/glm4/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}