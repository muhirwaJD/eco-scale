/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        eco: {
          green: "#22c55e",   // primary accent (brighter, modern)
          light: "#4ade80",   // chart line / positive text
          dark: "#16a34a",
          amber: "#fbbf24",
          red: "#f87171",     // HPA line / scale-down / danger
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(34,197,94,0.35), 0 10px 30px -10px rgba(34,197,94,0.45)",
      },
    },
  },
  plugins: [],
};
