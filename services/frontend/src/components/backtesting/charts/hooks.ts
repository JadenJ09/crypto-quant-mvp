/**
 * Chart hooks for accessing chart context
 */

import { useContext } from 'react';
import { ChartContext } from './ChartContainer';

// Hook to access chart context
export const useChart = () => {
    const context = useContext(ChartContext);
    if (!context) {
        throw new Error('useChart must be used within a ChartContainer');
    }
    return context;
};
