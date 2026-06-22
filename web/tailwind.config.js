/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        eco: {
          green: "#2E7D32",
          light: "#4CAF50",
          amber: "#F59E0B",
          red: "#EF4444",
        },
      },
    },
  },
  plugins: [],
};
