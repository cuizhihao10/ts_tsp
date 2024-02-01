<template>
    <div class="aside">
        <el-aside>
            <el-menu class="aside_main" unique-opened :collapse="isCollapse" default-active="1" text-color="#fff"
                active-text-color="#ccc" :collapse-transition="false"  @click="handleClick" @open="handleOpen"
                @close="handleClose">
                <a href="/" class="logo">
                    <img color="white" src="@/assets/icons/脑瘤.svg" alt="">
                    <h1 v-show="!isCollapse" style="min-width: 150px; font-size: 30px; height: 100%;">GMMAS</h1>
                </a>
                <el-menu-item  index="1" @click="goPage('/')">
                    <el-icon class="icon_size" style="color:rgb(189, 103, 5)">
                        <IEpHomeFilled />
                    </el-icon>
                    <span class="font_size">Homepage</span>
                </el-menu-item>
                <el-sub-menu class="sub-menu" index="2">
                    <template #title>
                        <el-icon class="icon_size" style="color:rgb(250, 0, 200);">
                            <IEpScissor />
                        </el-icon>
                        <span class="font_size">
                            Segmentation Task
                        </span>
                    </template>
                    <el-menu-item index="3" @click="goPage('/segmentation_history/')">
                        <el-icon class="icon_size">
                            <IEpList />
                        </el-icon>
                        <span class="font_size_sub">
                            Historical Tasks
                        </span>
                    </el-menu-item>
                </el-sub-menu>
                <el-sub-menu class="sub-menu" index="4">
                    <template #title>
                        <el-icon class="icon_size" style="color:rgb(13, 253, 0)">
                            <IEpGuide />
                        </el-icon>
                        <span class="font_size">
                            Classification Task
                        </span>
                    </template>
                    <el-menu-item index="5" @click="goPage('/classification_history/')">
                        <el-icon class="icon_size">
                            <IEpList />
                        </el-icon>
                        <span class="font_size_sub">
                            Historical Tasks
                        </span>
                    </el-menu-item>
                </el-sub-menu>
                <!-- <el-menu-item index="8">
                    <el-icon class="icon_size" style="color:rgb(7, 227, 247)">
                        <IEpChatLineRound />
                    </el-icon>
                    <span class="font_size">
                        AI Consulting
                    </span>
                </el-menu-item> -->
                <!-- <el-tag effect="dark" type="danger">
                    测试版
                </el-tag> -->
            </el-menu>

        </el-aside>
    </div>
</template>

<script setup lang='ts'>
import { isCollapse } from '@/store/modules/isCollapse'
import router from '@/router/index'

const is_active = ref('1');
const menuActive = router.currentRoute.value.path

const goPage = (path: string) => {
    router.push(path)
}

const handleClick = (key: string, keyPath: string[]) => {
    is_active.value = key;
}

const handleOpen = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
}
const handleClose = (key: string, keyPath: string[]) => {
    console.log(key, keyPath)
}
</script>

<style lang="scss" scoped>
.el-aside {
    background-color: rgb(29, 37, 51);
    color: rgb(223, 255, 208);
    height: 100vh;
    font-family: 'Franklin Gothic Medium', 'Arial Narrow', Arial, sans-serif;
    font-weight: 600;
}

.el-aside::-webkit-scrollbar {
    display: none;
}

.logo {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 8vh;
    text-decoration: none;

    img {
        width: 64px;
        height: 64px;
    }

    h1 {
        margin-left: 10px;
        font-size: 16px;
        color: #fff;
        text-align: center;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
}

.el-menu {
    background-color: rgb(29, 37, 51);
    border-right: none;
    width: 400px;

    // el-tag {
    //     position: absolute;
    //     bottom: 50px;
    //     left: 70px;
    //     margin-top: 50px;
    // }
    &.el-menu--collapse {
        width: 60px;
        & h1 {
            // min-width: 50px;
            display: none;
        }
    }
}

.el-sub-menu.sub-menu {
    height: 100px;
    transition: 0.3s ease-in;
}

.el-sub-menu> :deep(.el-sub-menu__title) {
    height: 100px;
    line-height: 100px;
    font-size: 20px;
    font-weight: 600;
    color: #fff;
    text-align: center;
    cursor: pointer;
}

.el-sub-menu.is-opened.sub-menu {
    height: 200px;
}

.el-sub-menu.is-opened> :deep(.el-sub-menu__title) {
    height: 100px;
}

.el-menu :hover {
    background-color: rgb(17, 22, 31) !important;
    color: #1ca352 !important;
}

.el-menu-item {
    background-color: rgb(29, 37, 51);
    height: 100px;
}

.el-sub-menu .el-menu-item {
    background-color: rgb(36, 47, 85);
    height: 100px;
}

.sub-menu :deep(.el-sub-menu__icon-arrow) {
    color: rgb(28, 163, 161) !important;
    font-size: large;
}

.el-menu-item.is-active {
    background: rgb(17, 22, 31) !important;
    color: #fff !important;
}

.el-submenu__title.is-active {
    background: #7f888f !important;
    color: #fff !important;
}

.font_size {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 25px;
}

.font_size_sub {
    font-family: Arial, Helvetica, sans-serif;
    font-size: 22px;
}

.icon_size {
    font-size: 26px;
}
</style>