# TIQ FreqUI Customization Guide

## Overview
This guide documents all customization points in the FreqUI source code for TIQ branding and future modifications.

## Directory Structure

```
frequi-main/
├── src/                    # Source code
│   ├── assets/            # Images and static assets
│   ├── components/        # Vue components
│   ├── views/            # Page views
│   ├── stores/           # State management
│   └── styles/           # Global styles
├── public/                # Public static files
├── index.html            # Base HTML template
└── package.json          # Project configuration
```

## Key Files for Branding

### 1. Logo Files
**Location:** `/src/assets/`
- `freqtrade-logo.png` - Main logo image (replace with TIQ logo)
- `freqtrade-logo-mask.png` - Logo mask for styling effects
- **Also:** `/public/favicon.ico` - Browser tab icon

### 2. Navigation Bar
**Location:** `/src/components/layout/NavBar.vue`
- Contains the top navigation bar with logo and menu
- Lines to modify:
  - Logo import: `import logo from '@/assets/freqtrade-logo.png'`
  - Brand text: Search for "Freqtrade" or "FreqUI"
  - Background color: Look for `bg-primary-500` or similar classes

### 3. Home Page
**Location:** `/src/views/HomeView.vue`
- Welcome message: "Welcome to the Freqtrade UI"
- Documentation links
- Team attribution: "wishes you the Freqtrade team"

### 4. Page Title
**Location:** `/index.html`
- `<title>FreqUI</title>` - Change to "TIQ Trading Platform"

### 5. Color Scheme
**Location:** `/src/stores/colors.ts`
- Defines color preferences for the entire application
- Key colors:
  - `colorUp`: Green/Red for price movements
  - `colorDown`: Red/Green for price movements
  - `colorProfit`: Profit color
  - `colorLoss`: Loss color

### 6. App Configuration
**Location:** `/package.json`
- Application name and metadata
- Build scripts

## Component Locations

### Main Layout Components

1. **NavBar** (`/src/components/layout/NavBar.vue`)
   - Top navigation bar
   - Bot selector dropdown
   - Menu items
   - User settings

2. **NavFooter** (`/src/components/layout/NavFooter.vue`)
   - Footer component (if present)

3. **BodyLayout** (`/src/components/layout/BodyLayout.vue`)
   - Main content wrapper
   - Layout structure

### Trading View Components

**Location:** `/src/views/TradingView.vue`
- Main trading interface
- Contains:
  - Trade list
  - Charts
  - Bot controls
  - Performance metrics

### Dashboard Components

**Location:** `/src/views/DashboardView.vue`
- Dashboard layout with draggable widgets
- Components used:
  - `/src/components/ftbot/BotProfit.vue` - Profit display
  - `/src/components/ftbot/BotBalance.vue` - Balance display
  - `/src/components/ftbot/BotStatus.vue` - Bot status
  - `/src/components/charts/CumProfitChart.vue` - Cumulative profit chart
  - `/src/components/charts/HourlyChart.vue` - Hourly profit chart

### Chart Components

**Location:** `/src/components/charts/`
- `CandleChart.vue` - Main candlestick chart
- `CumProfitChart.vue` - Cumulative profit chart
- `ProfitDistributionChart.vue` - Profit distribution
- `TradesLogChart.vue` - Trade history chart
- `BalanceChart.vue` - Balance over time

### Bot Control Components

**Location:** `/src/components/ftbot/`
- `BotControls.vue` - Start/Stop/Reload controls
- `ForceEntryForm.vue` - Manual trade entry
- `ForceExitForm.vue` - Manual trade exit
- `TradeList.vue` - List of open/closed trades
- `TradeDetail.vue` - Individual trade details

## Styling System

### Tailwind CSS
- Uses Tailwind CSS for utility classes
- Configuration: `/tailwind.config.js` (if present)
- Custom styles: `/src/styles/tailwind.css`

### Color Classes
Common color classes to modify:
- `bg-primary-500` - Primary background (currently red)
- `text-primary` - Primary text color
- `bg-gray-100 dark:bg-gray-900` - Light/dark mode backgrounds
- `border-primary` - Primary border color

### Theme System
- Dark/Light mode support
- Theme toggle in `/src/components/ThemeSelect.vue`

## Build Process

### Development
```bash
npm install        # Install dependencies
npm run dev       # Start development server
```

### Production Build
```bash
npm run build     # Build for production
# Output will be in /dist folder
```

### Deployment
After building, copy contents of `/dist` folder to:
`/Users/park/freqtrade/.venv/lib/python3.12/site-packages/freqtrade/rpc/api_server/ui/installed/`

## Customization Checklist

### For TIQ Branding:
- [ ] Replace `/src/assets/freqtrade-logo.png` with TIQ logo
- [ ] Replace `/src/assets/freqtrade-logo-mask.png` with TIQ logo mask
- [ ] Update `/public/favicon.ico` with TIQ icon
- [ ] Modify `/src/components/layout/NavBar.vue`:
  - [ ] Change "Freqtrade UI" to "TIQ Platform"
  - [ ] Update navbar background color to black
- [ ] Update `/src/views/HomeView.vue`:
  - [ ] Change welcome message to "Welcome to the TIQ Trading Platform"
  - [ ] Update documentation link text to "TIQ Documentation"
  - [ ] Change team attribution to "TIQ team"
- [ ] Update `/index.html` title to "TIQ Trading Platform"
- [ ] Modify color scheme in `/src/stores/colors.ts`
- [ ] Update any hardcoded "Freqtrade" references throughout the codebase

### Color Scheme (TIQ Black and Light Blue)
```css
--tiq-black: #000000
--tiq-dark: #0a0a0a
--tiq-light-blue: #5EEAD4
--tiq-blue: #3ABCD1
--tiq-dark-blue: #2A8FA6
```

## Advanced Customizations

### Rearranging Layout
1. **Trading View Layout**
   - Edit `/src/stores/layout.ts` for default positions
   - Modify grid layout in `/src/views/TradingView.vue`

2. **Dashboard Widgets**
   - Edit `/src/views/DashboardView.vue`
   - Adjust widget positions in the grid layout array

3. **Menu Structure**
   - Edit `/src/router/index.ts` for routes
   - Modify `/src/components/layout/NavBar.vue` for menu items

### Adding New Components
1. Create component in `/src/components/`
2. Import in parent view/component
3. Register in component's `components` object
4. Add to template

### API Integration
- API calls are in `/src/composables/api.ts`
- WebSocket handling in `/src/stores/ftbot.ts`

## Notes
- Always run `npm run build` after making changes
- Test in development with `npm run dev` first
- Keep backups of original files
- The UI uses Vue 3 with Composition API
- State management uses Pinia stores
- Charts use ECharts library