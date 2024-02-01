<template>
    <div class="theme_status">
        <div class="icon1">
            <Icon>
                <WbSunnyOutlined />
            </Icon>
        </div>
        <div class="switch" :class="useTheme.theme" @click="useTheme.setTheme">
        <div class="bar"></div>
        <div class="icon2">
            <Icon>
                <NightsStayFilled />
            </Icon>
        </div>
    </div>
    </div>
    
</template>

<script setup lang="ts">
import { useThemeStore } from "@/store/modules/useTheme";
// import { WbSunnyOutlined, NightsStayFilled } from "@vicons/material";
import { Icon } from '@vicons/utils';


const useTheme = useThemeStore();

</script>

<style scoped>
.theme_status {
    display: flex;
    position: relative;
    top: -9vh;
    left: 46vw;
    float: left;
}
.icon1 {
    position: absolute;
    width: 50px;
    height: 50px;
    /* justify-content: center; */
    /* align-items: center; */
    transition: 0.5s ease-in-out;
}
.switch {
    padding: 5px;
    transition: 0.2s ease-in-out;
    width: 120px;
    height: 50px;
    border-radius: 50px;
    /* text-align: center; */
    background: lightslategray;
    box-shadow: -4px 4px 15px inset rgba(0, 0, 0);
    cursor: pointer;
    position: relative;
}

/* .switch .icon2 {} */
</style>


