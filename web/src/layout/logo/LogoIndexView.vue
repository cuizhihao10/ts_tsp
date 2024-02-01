<template>
    <el-row class="logo">
        <el-col :span="6">
            <svg-icon name="脑瘤" width="50px" height="54px" color="white"></svg-icon>
        </el-col>
    </el-row>
</template>

<script setup lang='ts'>
import { isCollapse } from '@/store/modules/isCollapse'

</script>


<style scoped>
.logo {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    color:white;
    padding: 0px;
}
</style>