<template>
    <!-- <div>
    <svg-icon name="脑瘤" width="100px" height="100px"></svg-icon>
  </div> -->
    <router-view v-if="isRouter"></router-view>
</template>

<script setup lang="ts">
// import NarBar from './components/NarBar.vue';

import { nextTick, provide, ref } from "vue";
const isRouter = ref(true);
const reload = () => {
    isRouter.value = false;
    nextTick(() => {
        isRouter.value = true;
    });
};
provide("reload", reload);
</script>

<style lang="scss" scoped></style>
