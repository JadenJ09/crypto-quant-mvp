/**
 * Chart Utilities
 * Helper functions for chart operations and configurations
 */

import { 
    ChartTheme, 
    LIGHT_THEME, 
    DARK_THEME, 
    IndicatorCategory,
    CHART_CONSTANTS,
    getIndicatorCategory,
    getReferenceLines,
    OHLCVDataPoint,
    IndicatorDataPoint 
} from './types';

/**
 * Get appropriate chart theme based on mode
 */
export const getChartTheme = (isDarkMode: boolean): ChartTheme => {
    return isDarkMode ? DARK_THEME : LIGHT_THEME;
};

/**
 * Calculate optimal chart height based on indicators
 */
export const calculateOptimalHeight = (
    indicatorData: Record<string, IndicatorDataPoint[]>,
    includeVolume: boolean = true
): number => {
    let height = CHART_CONSTANTS.MAIN_PANE_HEIGHT;
    
    // Add volume pane height if volume data exists
    if (includeVolume) {
        height += CHART_CONSTANTS.VOLUME_PANE_HEIGHT;
    }
    
    // Count indicators that need separate panes
    const separatePaneIndicators = Object.keys(indicatorData).filter(name => {
        const category = getIndicatorCategory(name);
        return category !== IndicatorCategory.MAIN_CHART && indicatorData[name].length > 0;
    });
    
    // Add height for each separate pane
    const separatePaneCount = separatePaneIndicators.length;
    height += separatePaneCount * CHART_CONSTANTS.INDICATOR_PANE_HEIGHT;
    
    // Add separator heights
    const totalPanes = 1 + (includeVolume ? 1 : 0) + separatePaneCount;
    height += (totalPanes - 1) * CHART_CONSTANTS.SEPARATOR_HEIGHT;
    
    return height;
};

/**
 * Validate OHLCV data structure
 */
export const validateOHLCVData = (data: OHLCVDataPoint[]): {
    isValid: boolean;
    errors: string[];
    validData: OHLCVDataPoint[];
} => {
    const errors: string[] = [];
    const validData: OHLCVDataPoint[] = [];
    
    if (!Array.isArray(data)) {
        return {
            isValid: false,
            errors: ['Data must be an array'],
            validData: []
        };
    }
    
    data.forEach((point, index) => {
        const pointErrors: string[] = [];
        
        // Check required fields
        if (!point.time) pointErrors.push(`Missing time at index ${index}`);
        if (typeof point.open !== 'number' || isNaN(point.open)) pointErrors.push(`Invalid open at index ${index}`);
        if (typeof point.high !== 'number' || isNaN(point.high)) pointErrors.push(`Invalid high at index ${index}`);
        if (typeof point.low !== 'number' || isNaN(point.low)) pointErrors.push(`Invalid low at index ${index}`);
        if (typeof point.close !== 'number' || isNaN(point.close)) pointErrors.push(`Invalid close at index ${index}`);
        
        // Check logical constraints
        if (point.high < point.low) pointErrors.push(`High < Low at index ${index}`);
        if (point.high < Math.max(point.open, point.close)) pointErrors.push(`High < max(open,close) at index ${index}`);
        if (point.low > Math.min(point.open, point.close)) pointErrors.push(`Low > min(open,close) at index ${index}`);
        
        if (pointErrors.length === 0) {
            validData.push(point);
        } else {
            errors.push(...pointErrors);
        }
    });
    
    return {
        isValid: errors.length === 0,
        errors,
        validData
    };
};

/**
 * Validate indicator data structure
 */
export const validateIndicatorData = (
    data: Record<string, IndicatorDataPoint[]>
): {
    isValid: boolean;
    errors: string[];
    validData: Record<string, IndicatorDataPoint[]>;
} => {
    const errors: string[] = [];
    const validData: Record<string, IndicatorDataPoint[]> = {};
    
    Object.entries(data).forEach(([indicatorName, points]) => {
        if (!Array.isArray(points)) {
            errors.push(`Indicator ${indicatorName} data must be an array`);
            return;
        }
        
        const validPoints: IndicatorDataPoint[] = [];
        
        points.forEach((point, index) => {
            if (!point.time) {
                errors.push(`Missing time in ${indicatorName} at index ${index}`);
                return;
            }
            
            if (typeof point.value !== 'number' || isNaN(point.value)) {
                errors.push(`Invalid value in ${indicatorName} at index ${index}`);
                return;
            }
            
            validPoints.push(point);
        });
        
        if (validPoints.length > 0) {
            validData[indicatorName] = validPoints;
        }
    });
    
    return {
        isValid: errors.length === 0,
        errors,
        validData
    };
};

/**
 * Group indicators by category for better organization
 */
export const groupIndicatorsByCategory = (
    indicatorData: Record<string, IndicatorDataPoint[]>
): Record<IndicatorCategory, string[]> => {
    const groups: Record<IndicatorCategory, string[]> = {
        [IndicatorCategory.MAIN_CHART]: [],
        [IndicatorCategory.OSCILLATOR]: [],
        [IndicatorCategory.MOMENTUM]: [],
        [IndicatorCategory.VOLUME]: [],
        [IndicatorCategory.VOLATILITY]: [],
    };
    
    Object.keys(indicatorData).forEach(indicatorName => {
        if (indicatorData[indicatorName].length > 0) {
            const category = getIndicatorCategory(indicatorName);
            groups[category].push(indicatorName);
        }
    });
    
    return groups;
};

/**
 * Get indicator configuration for UI display
 */
export const getIndicatorDisplayConfig = (indicatorName: string) => {
    const category = getIndicatorCategory(indicatorName);
    const referenceLines = getReferenceLines(indicatorName);
    
    return {
        name: indicatorName,
        category,
        displayName: formatIndicatorName(indicatorName),
        separatePane: category !== IndicatorCategory.MAIN_CHART,
        hasReferenceLines: referenceLines.length > 0,
        referenceLines,
        defaultColor: getDefaultIndicatorColor(indicatorName),
    };
};

/**
 * Format indicator name for display
 */
export const formatIndicatorName = (indicatorName: string): string => {
    // Convert snake_case to Title Case
    return indicatorName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
};

/**
 * Get default color for specific indicators
 */
export const getDefaultIndicatorColor = (indicatorName: string): string => {
    const nameUpper = indicatorName.toUpperCase();
    
    if (nameUpper.includes('RSI')) return '#FF6D00';
    if (nameUpper.includes('MACD')) return '#2962FF';
    if (nameUpper.includes('SIGNAL')) return '#FF6D00';
    if (nameUpper.includes('EMA')) return '#2E7D32';
    if (nameUpper.includes('SMA')) return '#D50000';
    if (nameUpper.includes('BOLLINGER')) return '#5D4037';
    
    return CHART_CONSTANTS.DEFAULT_INDICATOR_COLORS[0];
};

/**
 * Performance optimization: Downsample data for large datasets
 */
export const downsampleData = <T extends { time: string }>(
    data: T[],
    maxPoints: number = 1000
): T[] => {
    if (data.length <= maxPoints) return data;
    
    const ratio = Math.ceil(data.length / maxPoints);
    const downsampled: T[] = [];
    
    for (let i = 0; i < data.length; i += ratio) {
        downsampled.push(data[i]);
    }
    
    // Always include the last point
    if (downsampled[downsampled.length - 1] !== data[data.length - 1]) {
        downsampled.push(data[data.length - 1]);
    }
    
    return downsampled;
};

/**
 * Time range utilities
 */
export const getTimeRange = (data: { time: string }[]): { start: string; end: string } | null => {
    if (data.length === 0) return null;
    
    const times = data.map(d => d.time).sort();
    return {
        start: times[0],
        end: times[times.length - 1]
    };
};

/**
 * Data alignment utility
 */
export const alignDataByTime = (
    ohlcvData: OHLCVDataPoint[],
    indicatorData: Record<string, IndicatorDataPoint[]>
): {
    alignedOHLCV: OHLCVDataPoint[];
    alignedIndicators: Record<string, IndicatorDataPoint[]>;
} => {
    if (ohlcvData.length === 0) {
        return {
            alignedOHLCV: [],
            alignedIndicators: {}
        };
    }
    
    // Get time range from OHLCV data
    const ohlcvTimes = new Set(ohlcvData.map(d => d.time));
    
    // Filter indicator data to match OHLCV timerange
    const alignedIndicators: Record<string, IndicatorDataPoint[]> = {};
    
    Object.entries(indicatorData).forEach(([name, data]) => {
        alignedIndicators[name] = data.filter(point => ohlcvTimes.has(point.time));
    });
    
    return {
        alignedOHLCV: ohlcvData,
        alignedIndicators
    };
};
