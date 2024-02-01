import { defineStore } from 'pinia'

interface UserInfoState {
    logined: boolean;
    uid: number;
    username: string;
}

const useUserInfoStore = defineStore('userinfo', () => {
    let userInfo = ref("")

    const getuserInfo = computed<UserInfoState>(() => {
        try {
            return JSON.parse(userInfo.value || window.localStorage.getItem('userinfo') || '{}')
        } catch (error) {
            ElMessage.error('获取用户信息失败')
            window.localStorage.setItem('userinfo', JSON.stringify({}))
            throw error;
        }
    })

    const setUserInfo = (userinfo: UserInfoState) => {
        userInfo.value = JSON.stringify(userinfo)
        window.localStorage.setItem('userinfo', userInfo.value)
    }

    const clearUserInfo = () => {
        userInfo.value = JSON.stringify({logined: false, uid: 0, username: ""})
        window.localStorage.setItem('userinfo', JSON.stringify({logined: false, uid: 0, username: ""}))
    }

    return { getuserInfo, setUserInfo, clearUserInfo }
})

export default useUserInfoStore
