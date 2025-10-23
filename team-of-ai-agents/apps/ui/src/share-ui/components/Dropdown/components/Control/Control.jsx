import React from "react";
import { components } from "react-select";
import Dialog from "../../../../components/Dialog/Dialog";
import Tooltip from "../../../../components/Tooltip/Tooltip";

const Control = props => {
  const { selectProps } = props;
  const control = <components.Control {...props} />;
  const controlRef = selectProps?.selectProps?.controlRef;
  if (controlRef)
    return (
      <Tooltip
        content={selectProps?.selectProps?.tooltipContent}
        hideTrigger={[Dialog.hideShowTriggers.MOUSE_LEAVE, Dialog.hideShowTriggers.CLICK]}
        showTrigger={[Dialog.hideShowTriggers.MOUSE_ENTER]}
      >
        <div className="l3-dropdown_scrollable-wrapper" ref={controlRef}>
          {control}
        </div>
      </Tooltip>
    );
  return control;
};

export default Control;
