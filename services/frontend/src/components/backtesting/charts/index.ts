/**
 * Chart Components Index
 * Clean exports for all chart-related components and utilities
 */

// Main Chart Component
export { default as BacktestChart } from './BacktestChart';
export type { BacktestChartProps } from './BacktestChart';

// Demo Components
// export { default as ChartDemo } from './ChartDemo';
// export { DynamicPaneDemo } from './DynamicPaneDemo';

// Utilities and Types
export * from './types';
export * from './utils';
export { PaneManager } from './PaneManager';

// Re-export commonly used types from lightweight-charts for convenience
export type { 
    IChartApi, 
    ISeriesApi, 
    SeriesMarker, 
    Time,
    CandlestickData,
    LineData,
    HistogramData 
} from 'lightweight-charts';
