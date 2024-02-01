<template>
<div class="common-layout">
    <el-container class="operation-wrapper">
        <el-aside :width="isCollapse ? '64px' : '400px'">
            <AsideNavigation />
        </el-aside>
        <el-container>
            <el-header>
                <HeaderNavigation />
            </el-header>
            <el-main>
                <router-view />
            </el-main>
        </el-container>
    </el-container>
</div>
</template>

<script setup lang='ts'>
import { isCollapse } from '@/store/modules/isCollapse'

</script>



<style lang="scss" scoped>
.operation-wrapper{
    height: 100vh;
}

.el-header {
    margin: 0px;
    padding: 0px;
    height: 8vh;
}

.el-aside {
    height: auto;
}

.el-main {
    height: auto;
    width: auto;
    background-color: rgb(255, 251, 231);
}

</style>