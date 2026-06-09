-- Improve PDF table column widths and wrapping for Token Highlighter report.
function Table(el)
  local n = #el.colspecs
  if n == 2 then
    el.colspecs[1] = {el.colspecs[1][1], 0.28}
    el.colspecs[2] = {el.colspecs[2][1], 0.72}
  elseif n == 3 then
    el.colspecs[1] = {el.colspecs[1][1], 0.16}
    el.colspecs[2] = {el.colspecs[2][1], 0.24}
    el.colspecs[3] = {el.colspecs[3][1], 0.60}
  end
  return el
end
