/** @type {import('jest').Config} */
const config = {
  projects: [
    {
      displayName: 'node',
      testEnvironment: 'node',
      transform: {
        '^.+\\.tsx?$': ['ts-jest', { tsconfig: { module: 'commonjs', jsx: 'react-jsx' } }],
      },
      moduleNameMapper: {
        '^@/(.*)$': '<rootDir>/src/$1',
      },
      testMatch: ['**/__tests__/**/*.test.ts'],
    },
    {
      displayName: 'jsdom',
      testEnvironment: 'jsdom',
      transform: {
        '^.+\\.tsx?$': ['ts-jest', { tsconfig: { module: 'commonjs', jsx: 'react-jsx' } }],
      },
      moduleNameMapper: {
        '^@/(.*)$': '<rootDir>/src/$1',
      },
      testMatch: ['**/__tests__/**/*.test.tsx'],
      setupFiles: ['<rootDir>/jest.setup.jsdom.js'],
    },
  ],
};

module.exports = config;
