import React from 'react'
import cx from 'classnames'
import { components } from 'react-select'
import { MENU_WRAPPER_CLASS_NAME } from '../../dropdown-constants'

const Menu = props => {
  const { children, Renderer, selectProps } = props
  const renderer = Renderer
  const withFixedPosition =
    selectProps?.selectProps?.insideOverflowContainer ||
    selectProps?.selectProps?.insideOverflowWithTransformContainer
  return (
    <components.Menu
      {...props}
      className={cx('menu', MENU_WRAPPER_CLASS_NAME, {
        ['dropdown-menu-wrapper--fixed-position']: withFixedPosition,
      })}
    >
      {Renderer && renderer(props)}
      {!Renderer && children}
    </components.Menu>
  )
}

export default Menu
