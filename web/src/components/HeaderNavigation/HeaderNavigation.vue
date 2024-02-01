<template>
    <el-header>
        <el-icon style="cursor: pointer; font-size: 28px;" @click="isCollapse=!isCollapse">
            <IEpExpand v-show="!isCollapse" />
            <IEpFold v-show="isCollapse" />
        </el-icon>

        <!-- <el-breadcrumb style="font-size: 25px;" separator="/">
            <el-breadcrumb-item :to="{ path: '/' }">homepage</el-breadcrumb-item>
            <el-breadcrumb-item>
                <a href="/">promotion management</a>
            </el-breadcrumb-item>
            <el-breadcrumb-item>promotion list</el-breadcrumb-item>
            <el-breadcrumb-item>promotion detail</el-breadcrumb-item>
        </el-breadcrumb> -->

        <el-dropdown>
            <span class="el-dropdown-link" >
                <svg-icon name="头像" width="84px" height="84px" color="white"></svg-icon>
                {{ userInfoStore.getuserInfo.username }}
                <el-icon class="el-icon--right">
                    <i-ep-arrow-down />
                </el-icon>
            </span>
            <template #dropdown>
            <el-dropdown-menu size="large" style="height: 50px; width: 150px; display: flex; justify-content: center; align-items: center;">
                <el-dropdown-item  style="font-size: 25px; height: 90%; width: 80%;display: flex; justify-content: center; align-items: center;" @click="handleLogout">Logout</el-dropdown-item>
            </el-dropdown-menu>
            </template>
        </el-dropdown>
    </el-header>


    <!-- <div class="top-nav-bar">
        <el-row >
            <el-col :span="12">
                <el-row>
                    <el-col :span="12">
                        <el-menu text-color="white" class="el-menu-bar" mode="horizontal" :ellipsis="false" :defaultActive="menuActive">
                            <el-menu-item index="3" @click="goPage('/dashboard/')">控制台</el-menu-item>
                            <el-menu-item index="1" @click="goPage('/table/')">我的算法</el-menu-item>
                            <el-menu-item index="2" @click="goPage('/404/')">应用市场</el-menu-item>
                            <el-menu-item index="4" @click="goPage('/project/')">项目列表</el-menu-item>
                        </el-menu>
                    </el-col>
                </el-row>
                <div class="app-title">
      
                </div>
            </el-col>
            <el-col :span="12">
                
            </el-col>
        </el-row>
    </div> -->
</template>

<script lang="ts" setup>
import router from '@/router/index'
import { isCollapse } from '@/store/modules/isCollapse'
import { logout } from '@/api/user'
import { useTokenStore } from '@/store/modules/mytoken'
import useUserInfoStore from '@/store/modules/userInfo';


const userInfoStore = useUserInfoStore()
const menuActive = router.currentRoute.value.path
// const goPage = (path: string) => {
//     router.push(path)
// }

const handleLogout = async() => {
    await ElMessageBox.confirm('Are you sure to exit?', 'Info', {
        confirmButtonText: 'Confirm',
        cancelButtonText: 'Cancel',
        type: 'warning'
    }).then(() => {
        logout().then(res => {
            if (res.data.code === 200) {
                userInfoStore.clearUserInfo()
                ElMessage.success('Exit successfully!')
                useTokenStore().saveToken({access_token: '', refresh_token: ''})
                router.push('/login/')
            } else {
                ElMessage.error('Exit failed!')
            }
        })
    }).catch(() => {
        ElMessage.info('Canceled exit!')
    })
}

console.log(menuActive)
</script>

<style lang="scss" scoped>
.el-header {
    width:auto;
    display: flex;
    align-items: center;
    background: linear-gradient(to left, rgb(163, 191, 240), rgb(29,37,51));
}

.el-icon {
    color:white;
    margin-left: 20px;
    margin-right: 20px;
}

/* 不被选中时的颜色 */
.el-breadcrumb :deep .el-breadcrumb__inner {
    color: #f2d6b2 !important;
    font-weight:400 !important;
}
/* 被选中时的颜色 */
.el-breadcrumb__item:last-child :deep .el-breadcrumb__inner {
    color: #9b6924 !important;
    font-weight:800 !important;
}

.el-dropdown {
    margin-left: auto;
    border: none;
    font-size: 25px;
}

.el-dropdown .el-dropdown-link {
    cursor: pointer;
    color: var(--el-color-white);
    display: flex;
    align-items: center;
    font-size: 25px;
}

:deep(.el-tooltip__trigger:focus-visible) {
    outline: unset;
}
</style>