<template>
    <svg :style="{width, height}" aria-hidden="true">
        <use :xlink:href="prefix + name" :fill="color"></use>
    </svg>
</template>

<script setup lang='ts'>

defineProps({
    //xlink:href属性值的前缀
    prefix: {
        type: String,
        default: '#icon-'
    },
    name: String,
    //接受父组件传递的颜色
    color: {
        type: String,
        default: ''
    },
    width: {
        type: String,
        default: '16px'
    },
    height: {
        type: String,
        default: '16px'
    }
})
</script>

<style scoped>

</style>