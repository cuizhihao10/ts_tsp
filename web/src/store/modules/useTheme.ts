import { ref, watchEffect } from 'vue'
import { defineStore } from 'pinia'

const LOCAL_KEY = '__theme__';
type Theme = 'light' | 'dark';
const theme = ref<Theme>(localStorage.getItem(LOCAL_KEY) as Theme || 'light');

watchEffect(() => {
    document.documentElement.dataset.theme = theme.value;
    localStorage.setItem(LOCAL_KEY, theme.value);
})

export const useThemeStore = defineStore('theme', () => {
    function setTheme() {
        theme.value = theme.value === 'dark' ? 'light' : 'dark';
    }
    return { theme, setTheme };
})
