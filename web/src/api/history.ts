import request from '@/api/axios'

export const seg_history = (data: any) => {
    return request({
        url: '/history/segmentation/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}

export const search_autocomplete = (data: object) => {
    return request({
        url: '/history/search_autocomplete/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}

export const seg_history_search = (data: object) => {
    return request({
        url: '/history/segmentation_search/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}

export const seg_history_detail = (data: object) => {
    return request({
        url: '/history/segmentation_detail/',
        method: 'GET',
        params: data,
    })
}


export const seg_history_delete = (data: any) => {
    return request({
        url: '/history/segmentation_delete/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}

export const classification_history = (data: any) => {
    return request({
        url: '/history/classification/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}

export const classification_history_detail = (data: object) => {
    return request({
        url: '/history/classification_detail/',
        method: 'GET',
        params: data,
    })
}

export const classification_history_delete = (data: any) => {
    return request({
        url: '/history/classification_delete/',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json;charset=UTF-8'
        },
        data,
    })
}
