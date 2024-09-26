const { nextui } = require("@nextui-org/react");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@nextui-org/theme/dist/**/*.{js,ts,jsx,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        primary: "#7828C8",
        secondary: "#7828C8",
        background: "#FFFFFF",
        foreground: "#7828C8",
        border: "#E2E8F0", // Added border color
      },
    },
  },
  darkMode: "class",
  plugins: [
    nextui({
      themes: {
        light: {
          colors: {
            primary: {
              DEFAULT: "#7828C8",
              foreground: "#FFFFFF",
            },
            focus: "#7828C8",
          },
        },
      },
    }),
  ],
}