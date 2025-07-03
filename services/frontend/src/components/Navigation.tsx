import React from 'react'
import { NavLink } from 'react-router-dom'
import { BarChart3, TrendingUp, Settings } from 'lucide-react'

const Navigation: React.FC = () => {
  const navLinks = [
    { to: '/', label: 'Dashboard', icon: BarChart3 },
    { to: '/backtesting', label: 'Backtesting', icon: TrendingUp },
    { to: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <nav className="flex space-x-1">
      {navLinks.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
              isActive
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-accent'
            }`
          }
        >
          <Icon size={16} />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}

export default Navigation
