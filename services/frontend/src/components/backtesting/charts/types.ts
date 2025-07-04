/**
 * Chart Types and Constants
 * Centralized definitions for chart data structures and configurations
 */

import { UTCTimestamp } from 'lightweight-charts';

// === CHART DATA TYPES ===
export interface OHLCVDataPoint {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface IndicatorDataPoint {
    time: string;
    value: number;
}

export interface TradeDataPoint {
    time: string;
    side: 'Buy' | 'Sell';
    price: number;
    quantity: number;
    exit_time?: string;
    exit_price?: number;
}

// === PANE CONFIGURATION ===
export interface IndicatorPaneConfig {
    paneType: 'price' | 'oscillator' | 'momentum' | 'volume' | 'volatility' | 'trend';
    range?: [number, number] | 'auto';
    referenceLines?: number[];
    height?: number;
    target: 'main' | 'separate'; // main = overlay on price chart, separate = own pane
    title?: string;
    description?: string;
    paneIndex?: number; // Official pane index for TradingView API
}

export interface DynamicPaneConfig {
    id: string;
    title: string;
    indicators: string[];
    height: number;
    paneType: 'price' | 'oscillator' | 'momentum' | 'volume' | 'volatility' | 'trend';
    paneIndex: number; // Official pane index
    referenceLines?: Array<{
        value: number;
        color: string;
        style?: number;
        label?: string;
    }>;
}

// === CHART CONFIGURATION ===
export interface ChartTheme {
    background: string;
    textColor: string;
    gridColor: string;
    upColor: string;
    downColor: string;
    wickUpColor: string;
    wickDownColor: string;
    volumeUpColor: string;
    volumeDownColor: string;
    buyMarkerColor: string;
    sellMarkerColor: string;
    indicatorColors: string[];
    rsiLineColor: string;
    rsiLevelColor: string;
    separatorColor: string;
    separatorHoverColor: string;
}

export interface PaneConfiguration {
    id: string;
    height: number;
    indicators: string[];
    showReferenceLines?: boolean;
    referenceLines?: Array<{
        value: number;
        color: string;
        style?: number; // 0=solid, 1=dotted, 2=dashed, 3=large dashed
        label?: string;
    }>;
}

// === INDICATOR CATEGORIES ===
export enum IndicatorCategory {
    MAIN_CHART = 'main_chart',      // SMA, EMA, Bollinger Bands
    OSCILLATOR = 'oscillator',      // RSI, Stochastic, Williams %R
    MOMENTUM = 'momentum',          // MACD, ROC, CMO
    VOLUME = 'volume',              // Volume, OBV, A/D Line
    VOLATILITY = 'volatility'       // ATR, Volatility
}

// === INDICATOR PANE CONFIGURATION DICTIONARY ===
export const INDICATOR_PANE_CONFIG: Record<string, IndicatorPaneConfig> = {
    // Price-related indicators - overlay on main price chart
    'sma': { 
        paneType: 'price',
        target: 'main',
        height: 0, // No extra height - overlays on main chart
        title: 'Simple Moving Average',
        description: 'Trend following indicator - overlays on price chart'
    },
    'ema': { 
        paneType: 'price',
        target: 'main',
        height: 0, // No extra height - overlays on main chart
        title: 'Exponential Moving Average',
        description: 'Trend following indicator - overlays on price chart'
    },
    'wma': { 
        paneType: 'price',
        target: 'main',
        height: 0, // No extra height - overlays on main chart
        title: 'Weighted Moving Average',
        description: 'Trend following indicator - overlays on price chart'
    },
    'bbands': { 
        paneType: 'price',
        target: 'main',
        height: 0, // No extra height - overlays on main chart
        title: 'Bollinger Bands',
        description: 'Volatility bands around moving average - overlays on price chart'
    },
    'sar': { 
        paneType: 'price',
        target: 'main',
        height: 0, // No extra height - overlays on main chart
        title: 'Parabolic SAR',
        description: 'Stop and reverse indicator - overlays on price chart'
    },
    
    // Oscillators - separate panes with 0-100 range
    'rsi': { 
        paneType: 'oscillator',
        range: [0, 100],
        referenceLines: [30, 50, 70],
        height: 150,
        target: 'separate',
        title: 'RSI',
        description: 'Relative Strength Index (0-100)'
    },
    'stoch': { 
        paneType: 'oscillator',
        range: [0, 100],
        referenceLines: [20, 80],
        height: 150,
        target: 'separate',
        title: 'Stochastic',
        description: 'Stochastic Oscillator (0-100)'
    },
    'stochf': { 
        paneType: 'oscillator',
        range: [0, 100],
        referenceLines: [20, 80],
        height: 150,
        target: 'separate',
        title: 'Fast Stochastic'
    },
    'stochrsi': { 
        paneType: 'oscillator',
        range: [0, 100],
        referenceLines: [20, 80],
        height: 150,
        target: 'separate',
        title: 'Stochastic RSI'
    },
    'willr': { 
        paneType: 'oscillator',
        range: [-100, 0],
        referenceLines: [-80, -20],
        height: 150,
        target: 'separate',
        title: 'Williams %R',
        description: 'Williams Percent Range (-100 to 0)'
    },
    'cci': { 
        paneType: 'oscillator',
        range: 'auto',
        referenceLines: [-100, 0, 100],
        height: 150,
        target: 'separate',
        title: 'CCI',
        description: 'Commodity Channel Index'
    },
    'mfi': { 
        paneType: 'oscillator',
        range: [0, 100],
        referenceLines: [20, 80],
        height: 150,
        target: 'separate',
        title: 'Money Flow Index'
    },
    
    // Momentum indicators - separate panes
    'macd': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'MACD',
        description: 'Moving Average Convergence Divergence'
    },
    'macd_line': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'MACD',
        description: 'MACD Line'
    },
    'macd_signal': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'MACD',
        description: 'MACD Signal Line'
    },
    'macd_histogram': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'MACD',
        description: 'MACD Histogram'
    },
    'roc': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'Rate of Change'
    },
    'mom': {
        paneType: 'momentum',
        range: 'auto',
        referenceLines: [0],
        height: 150,
        target: 'separate',
        title: 'Momentum'
    },
    'adx': {
        paneType: 'momentum',
        range: [0, 100],
        referenceLines: [25, 50],
        height: 150,
        target: 'separate',
        title: 'ADX',
        description: 'Average Directional Index'
    },
    
    // Volume indicators - separate panes
    'obv': {
        paneType: 'volume',
        range: 'auto',
        height: 150,
        target: 'separate',
        title: 'On-Balance Volume'
    },
    'ad': {
        paneType: 'volume',
        range: 'auto',
        height: 150,
        target: 'separate',
        title: 'Accumulation/Distribution'
    },
    
    // Volatility indicators - separate panes
    'atr': {
        paneType: 'volatility',
        range: 'auto',
        height: 150,
        target: 'separate',
        title: 'Average True Range'
    },
    'stddev': {
        paneType: 'volatility',
        range: 'auto',
        height: 150,
        target: 'separate',
        title: 'Standard Deviation'
    }
};

// === CHART CONSTANTS ===
export const CHART_CONSTANTS = {
    DEFAULT_MAIN_PANE_HEIGHT: 400,     // Fixed height for main price chart
    MIN_MAIN_PANE_HEIGHT: 400,         // Minimum height for main price chart
    INDICATOR_PANE_HEIGHT: 150,        // Fixed height for each indicator subpane
    VOLUME_PANE_HEIGHT: 150,           // Fixed height for volume pane
    MIN_PANE_HEIGHT: 150,
    SEPARATOR_HEIGHT: 4,
    
    // RSI Reference Lines
    RSI_OVERBOUGHT: 70,
    RSI_OVERSOLD: 30,
    RSI_MIDLINE: 50,
    
    // Stochastic Reference Lines
    STOCH_OVERBOUGHT: 80,
    STOCH_OVERSOLD: 20,
    STOCH_MIDLINE: 50,
    
    // Williams %R Reference Lines
    WILLR_OVERBOUGHT: -20,
    WILLR_OVERSOLD: -80,
    WILLR_MIDLINE: -50,
    
    // CCI Reference Lines
    CCI_OVERBOUGHT: 100,
    CCI_OVERSOLD: -100,
    CCI_ZERO: 0,
    
    // Default indicator colors
    DEFAULT_INDICATOR_COLORS: [
        '#2962FF', // Blue
        '#FF6D00', // Orange
        '#2E7D32', // Green
        '#D50000', // Red
        '#5D4037', // Brown
        '#7B1FA2', // Purple
        '#C2185B', // Pink
        '#00796B', // Teal
        '#F57C00', // Amber
        '#455A64', // Blue Grey
    ],
};

// === CHART THEMES ===
export const LIGHT_THEME: ChartTheme = {
    background: '#FFFFFF',
    textColor: '#333333',
    gridColor: '#E6E6E6',
    upColor: '#26A69A',
    downColor: '#EF5350',
    wickUpColor: '#26A69A',
    wickDownColor: '#EF5350',
    volumeUpColor: 'rgba(38, 166, 154, 0.5)',
    volumeDownColor: 'rgba(239, 83, 80, 0.5)',
    buyMarkerColor: '#2196F3',
    sellMarkerColor: '#FF9800',
    indicatorColors: CHART_CONSTANTS.DEFAULT_INDICATOR_COLORS,
    rsiLineColor: '#FF6D00',
    rsiLevelColor: '#787B86',
    separatorColor: '#E6E6E6',
    separatorHoverColor: '#CCCCCC',
};

export const DARK_THEME: ChartTheme = {
    background: 'hsl(200, 25%, 8%)',
    textColor: 'hsl(180, 10%, 95%)',
    gridColor: 'hsl(200, 20%, 25%)',
    upColor: '#26A69A',
    downColor: '#EF5350',
    wickUpColor: '#26A69A',
    wickDownColor: '#EF5350',
    volumeUpColor: 'rgba(38, 166, 154, 0.5)',
    volumeDownColor: 'rgba(239, 83, 80, 0.5)',
    buyMarkerColor: '#58A6FF',
    sellMarkerColor: '#FFA657',
    indicatorColors: ['#58A6FF', '#F78166', '#7EE787', '#FF7B72', '#FFA657', '#A5A5FF', '#FF9F9F', '#9FFF9F', '#FFFF9F', '#9FFFFF'],
    rsiLineColor: '#F78166',
    rsiLevelColor: 'hsl(180, 10%, 65%)',
    separatorColor: 'hsl(200, 20%, 25%)',
    separatorHoverColor: 'hsl(200, 20%, 35%)',
};

// === UTILITY FUNCTIONS ===
export const convertTime = (timeStr: string): UTCTimestamp => {
    return (new Date(timeStr).getTime() / 1000) as UTCTimestamp;
};

export const getIndicatorPaneConfig = (indicatorName: string): IndicatorPaneConfig => {
    const nameKey = indicatorName.toLowerCase().replace(/[^a-z0-9_]/g, '');
    return INDICATOR_PANE_CONFIG[nameKey] || {
        paneType: 'trend',
        target: 'separate',
        height: 120,
        title: indicatorName.toUpperCase()
    };
};

export const getIndicatorCategory = (indicatorName: string): IndicatorCategory => {
    const config = getIndicatorPaneConfig(indicatorName);
    switch (config.paneType) {
        case 'price':
        case 'trend':
            return IndicatorCategory.MAIN_CHART;
        case 'oscillator':
            return IndicatorCategory.OSCILLATOR;
        case 'momentum':
            return IndicatorCategory.MOMENTUM;
        case 'volume':
            return IndicatorCategory.VOLUME;
        case 'volatility':
            return IndicatorCategory.VOLATILITY;
        default:
            return IndicatorCategory.MAIN_CHART;
    }
};

export const shouldUseSeparatePane = (indicatorName: string): boolean => {
    // Check if indicator should use separate pane based on configuration
    const config = getIndicatorPaneConfig(indicatorName);
    return config.target === 'separate';
};

export const createDynamicPaneConfig = (indicators: string[]): DynamicPaneConfig[] => {
    // Only process indicators that use separate panes
    const separatePaneIndicators = indicators.filter(indicator => {
        const config = getIndicatorPaneConfig(indicator);
        return config.target === 'separate';
    });
    
    // Group related indicators (like MACD components) together
    const groupedIndicators = groupRelatedIndicators(separatePaneIndicators);
    
    // Create one pane per group (ensuring MACD components are in single pane)
    return groupedIndicators.map((indicatorGroup, index) => {
        const firstIndicator = indicatorGroup[0];
        const config = getIndicatorPaneConfig(firstIndicator);
        const paneType = config.paneType as 'price' | 'oscillator' | 'momentum' | 'volume' | 'volatility' | 'trend';
        const baseIndicatorName = getBaseIndicatorName(firstIndicator);
        
        return {
            id: `${baseIndicatorName}-pane-${index}`,
            title: config.title || baseIndicatorName.toUpperCase(),
            indicators: indicatorGroup,
            height: 150, // Fixed 150px height for each pane
            paneType,
            paneIndex: index + 1, // Start from pane 1 (pane 0 is main price chart)
            referenceLines: config.referenceLines?.map(value => ({
                value,
                color: value === 0 ? '#888888' : value > 50 ? '#ff4444' : '#44ff44',
                style: 2,
                label: `${value}`
            }))
        };
    });
};

// Helper function to group related indicators (like MACD components)
const groupRelatedIndicators = (indicators: string[]): string[][] => {
    const groups: string[][] = [];
    const processed = new Set<string>();
    
    console.log('🔍 Grouping indicators:', indicators);
    
    indicators.forEach(indicator => {
        if (processed.has(indicator)) return;
        
        const baseIndicatorName = getBaseIndicatorName(indicator);
        const relatedIndicators = indicators.filter(ind => 
            getBaseIndicatorName(ind) === baseIndicatorName
        );
        
        console.log(`📊 Base indicator "${baseIndicatorName}" groups these indicators:`, relatedIndicators);
        
        groups.push(relatedIndicators);
        relatedIndicators.forEach(ind => processed.add(ind));
    });
    
    console.log('✅ Final indicator groups:', groups);
    return groups;
};

// Helper function to get base indicator name (e.g., "macd" from "macd_line")
const getBaseIndicatorName = (indicatorName: string): string => {
    const lowerName = indicatorName.toLowerCase();
    
    console.log(`🔍 Getting base name for: "${indicatorName}" -> "${lowerName}"`);
    
    // Handle MACD variations - all should be grouped together
    if (lowerName.includes('macd') || lowerName.includes('signal') || lowerName.includes('histogram')) {
        console.log(`✅ Grouping "${indicatorName}" as MACD`);
        return 'macd';
    }
    
    // Handle Stochastic variations
    if (lowerName.includes('stoch')) {
        console.log(`✅ Grouping "${indicatorName}" as STOCH`);
        return 'stoch';
    }
    
    // Handle Bollinger Bands variations
    if (lowerName.includes('bb') || lowerName.includes('bollinger')) {
        console.log(`✅ Grouping "${indicatorName}" as BB`);
        return 'bb';
    }
    
    // Handle RSI variations (RSI_14, RSI_21, etc.)
    if (lowerName.includes('rsi')) {
        console.log(`✅ Grouping "${indicatorName}" as RSI`);
        return 'rsi';
    }
    
    // Handle SMA variations (SMA_20, SMA_50, etc.)
    if (lowerName.includes('sma')) {
        console.log(`✅ Grouping "${indicatorName}" as SMA`);
        return 'sma';
    }
    
    // Handle EMA variations (EMA_21, EMA_55, etc.)
    if (lowerName.includes('ema')) {
        console.log(`✅ Grouping "${indicatorName}" as EMA`);
        return 'ema';
    }
    
    // Handle other compound indicators by taking the first part
    if (lowerName.includes('_')) {
        const baseName = lowerName.split('_')[0];
        console.log(`✅ Grouping "${indicatorName}" as "${baseName}" (split by underscore)`);
        return baseName;
    }
    
    console.log(`✅ Using full name for "${indicatorName}"`);
    return lowerName;
};

export const getReferenceLines = (indicatorName: string): Array<{ value: number; color: string; style?: number; label?: string }> => {
    const config = getIndicatorPaneConfig(indicatorName);
    
    if (!config.referenceLines) return [];
    
    return config.referenceLines.map(value => {
        let color = '#888888';
        let label = `${value}`;
        
        // Color coding based on common indicator ranges
        if (config.paneType === 'oscillator') {
            if (value <= 30 || value <= 20) {
                color = '#44ff44'; // Green for oversold
                label = `Oversold (${value})`;
            } else if (value >= 70 || value >= 80) {
                color = '#ff4444'; // Red for overbought
                label = `Overbought (${value})`;
            } else if (value === 50) {
                color = '#888888'; // Gray for midline
                label = `Midline (${value})`;
            }
        } else if (config.paneType === 'momentum') {
            if (value === 0) {
                color = '#888888';
                label = 'Zero Line';
            }
        }
        
        return {
            value,
            color,
            style: 2, // Dashed line
            label
        };
    });
};
