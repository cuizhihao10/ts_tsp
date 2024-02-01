import { defineStore } from 'pinia'

interface Token { 
    access_token: string;
    refresh_token: string;
}

export const useTokenStore = defineStore("mytoken", () => {
    const tokenJson = ref("")
    const token = computed<Token>(() => {
        try {
            return JSON.parse(tokenJson.value || window.localStorage.getItem('token') || '{}')
        } catch (error) {
            ElMessage.error('token解析失败')
            window.localStorage.setItem('token', JSON.stringify({}))
            throw error;
        }
    })

    const saveToken = (token: Token) => {
        tokenJson.value = JSON.stringify(token)
        window.localStorage.setItem('token', JSON.stringify(token))
    }

    const clearToken = () => {
        const refresh_token = token.value.refresh_token;
        tokenJson.value = JSON.stringify({ access_token: "", refresh_token: refresh_token })
        window.localStorage.setItem('token', JSON.stringify({ access_token: "", refresh_token: refresh_token }))
    }

    const refreah_access_token = (new_token: string) => {
        const old_token = token.value
        saveToken({ access_token: new_token, refresh_token: old_token.refresh_token })
    }

    return { token, saveToken, clearToken, refreah_access_token }
})
