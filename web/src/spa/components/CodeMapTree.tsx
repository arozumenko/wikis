/**
 * CodeMapTree — Compact explorer-style tree view for Code Map chat mode.
 *
 * Layout (like VS Code file explorer):
 *   ▼  auth.py  ·  backend/app/auth.py        3 sym
 *   │    ƒ  _fetch_jwks      method   :88
 *   │    ƒ  _validate_pat    method   :112  → client [calls]
 *   ▶  routes.py  ·  backend/app/api/routes.py  2 sym
 */
import React, { useState } from 'react';
import { Box, Stack, Tooltip, Typography, useTheme } from '@mui/material';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import InsertDriveFileOutlinedIcon from '@mui/icons-material/InsertDriveFileOutlined';
import AccountTreeOutlinedIcon from '@mui/icons-material/AccountTreeOutlined';
import type { components } from '../api/types.generated';

type CodeMapData = components['schemas']['CodeMapData'];
type CodeMapSection = components['schemas']['CodeMapSection'];
type CodeMapSymbol = components['schemas']['CodeMapSymbol'];
type CallStack = components['schemas']['CallStack'];
type CallStackStep = components['schemas']['CallStackStep'];

interface Props {
  data: CodeMapData;
}

/* ── symbol-type label → colour token ─────────────────────────── */
const TYPE_COLOR: Record<string, string> = {
  function:    'primary.main',
  method:      'primary.main',
  class:       'secondary.main',
  interface:   'info.main',
  type:        'info.main',
  constant:    'warning.main',
  variable:    'text.secondary',
  parameter:   'text.secondary',
  constructor: 'secondary.main',
};

/* Short mono label shown instead of a chip */
const TYPE_ABBR: Record<string, string> = {
  function:    'fn',
  method:      'fn',
  class:       'cls',
  interface:   'if',
  type:        'ty',
  constant:    'con',
  variable:    'var',
  parameter:   'par',
  constructor: 'ctor',
};

/* ── SymbolRow ──────────────────────────────────────────────────── */
function SymbolRow({ sym }: { sym: CodeMapSymbol }) {
  const theme = useTheme();
  const typeKey = (sym.symbol_type ?? '').toLowerCase();
  const abbr  = TYPE_ABBR[typeKey] ?? (typeKey.slice(0, 3) || '·');
  const color = TYPE_COLOR[typeKey] ?? 'text.secondary';
  const lineRef = sym.line_start != null ? `:${sym.line_start}` : '';
  const rels = sym.relationships ?? [];

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'baseline',
        gap: 0.75,
        pl: '36px',       /* indent past the tree guide line */
        pr: 1,
        py: '3px',
        minHeight: 24,
        position: 'relative',
        '&:hover': { bgcolor: 'action.hover', borderRadius: 1 },
        /* vertical guide line from parent file row */
        '&::before': {
          content: '""',
          position: 'absolute',
          left: 16,
          top: 0,
          bottom: 0,
          width: '1px',
          bgcolor: 'divider',
        },
        /* horizontal connector tick */
        '&::after': {
          content: '""',
          position: 'absolute',
          left: 16,
          top: '50%',
          width: '10px',
          height: '1px',
          bgcolor: 'divider',
        },
      }}
    >
      {/* type abbr badge */}
      <Typography
        component="span"
        sx={{
          fontSize: '0.6rem',
          fontFamily: 'monospace',
          fontWeight: 700,
          color,
          minWidth: 26,
          flexShrink: 0,
          lineHeight: 1,
        }}
      >
        {abbr}
      </Typography>

      {/* symbol name + description */}
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="caption"
          fontWeight={600}
          noWrap
          sx={{ lineHeight: 1.4 }}
        >
          {sym.name}
        </Typography>
        {sym.description && (
          <Typography
            variant="caption"
            color="text.secondary"
            noWrap
            sx={{ display: 'block', fontSize: '0.6rem', lineHeight: 1.3, mt: '-1px' }}
          >
            {sym.description}
          </Typography>
        )}
      </Box>

      {/* file:line — only if different from parent section */}
      {lineRef && (
        <Typography variant="caption" color="text.disabled" sx={{ flexShrink: 0, fontSize: '0.6rem' }}>
          {lineRef}
        </Typography>
      )}

      {/* relationship pills — up to 2, rest in tooltip */}
      {rels.length > 0 && (
        <Tooltip
          title={rels.join('\n')}
          placement="left"
          slotProps={{ tooltip: { sx: { whiteSpace: 'pre-line', maxWidth: 320 } } }}
        >
          <Box sx={{ display: 'flex', gap: 0.25, flexShrink: 0, alignItems: 'center' }}>
            {rels.slice(0, 2).map((rel, ri) => {
              const arrow = rel.includes('→') ? '→' : rel.includes('←') ? '←' : '';
              const parts = rel.split(/[→←]/).map(s => s.trim());
              const label = arrow
                ? `${arrow} ${parts[1]?.split(' ')[0] ?? ''}`.trim()
                : rel.slice(0, 16);
              return (
                <Box
                  key={ri}
                  sx={{
                    fontSize: '0.55rem',
                    px: 0.5,
                    py: '1px',
                    borderRadius: 0.5,
                    bgcolor: theme.palette.mode === 'dark'
                      ? 'rgba(255,255,255,0.08)'
                      : 'rgba(0,0,0,0.06)',
                    color: 'text.secondary',
                    fontFamily: 'monospace',
                    lineHeight: 1.4,
                    whiteSpace: 'nowrap',
                    maxWidth: 80,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {label}
                </Box>
              );
            })}
            {rels.length > 2 && (
              <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.55rem' }}>
                +{rels.length - 2}
              </Typography>
            )}
          </Box>
        </Tooltip>
      )}
    </Box>
  );
}

/* ── SectionRow (file node) ─────────────────────────────────────── */
function _SectionRow({ section, sectionIdx, defaultExpanded }: {
  section: CodeMapSection;
  sectionIdx: number;
  defaultExpanded: boolean;
}) {
  const [open, setOpen] = useState(defaultExpanded);
  const syms = section.symbols ?? [];
  const fileBase = section.title || (section.file_path ? section.file_path.split('/').pop() : '—') || '—';
  const fileFull = section.file_path || '';
  const isLong  = fileFull.length > fileBase.length;

  return (
    <>
      {/* File row */}
      <Box
        onClick={() => setOpen(v => !v)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          px: 0.5,
          py: '5px',
          cursor: 'pointer',
          borderRadius: 1,
          userSelect: 'none',
          '&:hover': { bgcolor: 'action.hover' },
        }}
      >
        {/* expand toggle */}
        <Box sx={{ display: 'flex', alignItems: 'center', color: 'text.secondary', flexShrink: 0 }}>
          {open
            ? <ExpandMoreIcon sx={{ fontSize: 16 }} />
            : <ChevronRightIcon sx={{ fontSize: 16 }} />}
        </Box>

        {/* section index bubble */}
        <Box
          sx={{
            minWidth: 18,
            height: 18,
            borderRadius: '50%',
            bgcolor: 'primary.main',
            color: 'primary.contrastText',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '0.6rem',
            fontWeight: 700,
            flexShrink: 0,
          }}
        >
          {sectionIdx + 1}
        </Box>

        <InsertDriveFileOutlinedIcon sx={{ fontSize: 14, color: 'text.secondary', flexShrink: 0 }} />

        {/* filename + full path */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Stack direction="row" alignItems="baseline" gap={0.5} flexWrap="nowrap">
            <Typography variant="caption" fontWeight={700} noWrap sx={{ lineHeight: 1.4 }}>
              {fileBase}
            </Typography>
            {isLong && (
              <Typography variant="caption" color="text.disabled" noWrap sx={{ fontSize: '0.6rem', lineHeight: 1.4 }}>
                · {fileFull}
              </Typography>
            )}
          </Stack>
        </Box>

        {/* symbol count */}
        <Typography
          variant="caption"
          color="text.disabled"
          sx={{ flexShrink: 0, fontSize: '0.6rem', pr: 0.5 }}
        >
          {syms.length} sym
        </Typography>
      </Box>

      {/* Symbol rows (children) */}
      {open && syms.map(sym => (
        <SymbolRow key={sym.id} sym={sym} />
      ))}
    </>
  );
}

/* ── CallStackView — vertical flow diagram ─────────────────────── */
function CallStackView({ stack }: { stack: CallStack }) {
  const theme = useTheme();
  const [open, setOpen] = useState(true);
  const steps = stack.steps ?? [];
  const isDark = theme.palette.mode === 'dark';

  return (
    <Box sx={{ mb: 1.5 }}>
      {/* Stack title */}
      <Box
        onClick={() => setOpen(v => !v)}
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          px: 0.5,
          py: '4px',
          cursor: 'pointer',
          borderRadius: 1,
          userSelect: 'none',
          '&:hover': { bgcolor: 'action.hover' },
        }}
      >
        {open
          ? <ExpandMoreIcon sx={{ fontSize: 14, color: 'text.secondary' }} />
          : <ChevronRightIcon sx={{ fontSize: 14, color: 'text.secondary' }} />}
        <Typography variant="caption" fontWeight={700} sx={{ lineHeight: 1.4 }}>
          {stack.title}
        </Typography>
        <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.6rem' }}>
          {steps.length} step{steps.length !== 1 ? 's' : ''}
        </Typography>
      </Box>

      {/* Steps */}
      {open && steps.map((step: CallStackStep, i: number) => (
        <Box key={i} sx={{ display: 'flex', pl: '20px', position: 'relative' }}>
          {/* Vertical connector line */}
          <Box sx={{
            position: 'absolute',
            left: 26,
            top: 0,
            bottom: i < steps.length - 1 ? 0 : '50%',
            width: '1.5px',
            bgcolor: isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.12)',
          }} />

          {/* Step content */}
          <Box sx={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: 0.75,
            py: '4px',
            pl: '18px',
            position: 'relative',
          }}>
            {/* Arrow dot */}
            <Box sx={{
              position: 'absolute',
              left: 10,
              top: 10,
              width: 7,
              height: 7,
              borderRadius: '50%',
              bgcolor: i === 0 ? 'primary.main' : isDark ? 'rgba(255,255,255,0.3)' : 'rgba(0,0,0,0.25)',
              flexShrink: 0,
              zIndex: 1,
            }} />

            <Box sx={{ minWidth: 0 }}>
              {/* Symbol name + file */}
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
                <Typography
                  variant="caption"
                  fontWeight={600}
                  sx={{ fontFamily: 'monospace', fontSize: '0.68rem', lineHeight: 1.4 }}
                >
                  {step.symbol}
                </Typography>
                {step.file_path && (
                  <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.55rem' }}>
                    {step.file_path.split('/').pop()}
                  </Typography>
                )}
              </Box>
              {/* Description */}
              {step.description && (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: 'block', fontSize: '0.6rem', lineHeight: 1.3, mt: '-1px' }}
                >
                  {step.description}
                </Typography>
              )}
            </Box>
          </Box>

          {/* Down arrow between steps */}
          {i < steps.length - 1 && (
            <Typography sx={{
              position: 'absolute',
              left: 23,
              bottom: -4,
              fontSize: '0.6rem',
              color: 'text.disabled',
              zIndex: 2,
            }}>
              ↓
            </Typography>
          )}
        </Box>
      ))}
    </Box>
  );
}

/* ── Root ───────────────────────────────────────────────────────── */
export default function CodeMapTree({ data }: Props) {
  const callStacks = data.call_stacks ?? [];
  const sections = data.sections ?? [];
  const hasContent = callStacks.length > 0 || sections.length > 0;

  if (!hasContent) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <AccountTreeOutlinedIcon sx={{ fontSize: 40, color: 'text.disabled', mb: 1 }} />
        <Typography variant="body2" color="text.secondary">
          No code structure found for this query.
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        height: '100%',
        overflow: 'auto',
        px: 1,
        pt: 1,
        pb: 2,
        /* Thin scrollbar */
        '&::-webkit-scrollbar': { width: 4 },
        '&::-webkit-scrollbar-thumb': { bgcolor: 'divider', borderRadius: 2 },
      }}
    >
      {/* Header */}
      <Stack direction="row" spacing={0.75} alignItems="center" px={0.5} mb={1}>
        <AccountTreeOutlinedIcon sx={{ fontSize: 15, color: 'primary.main' }} />
        <Typography variant="caption" fontWeight={700} color="text.primary">
          Code Map
        </Typography>
        <Typography variant="caption" color="text.disabled">
          {callStacks.length} flow{callStacks.length !== 1 ? 's' : ''}
        </Typography>
      </Stack>

      {/* Call-flow summary */}
      {data.summary && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: 'block', px: 0.5, mb: 1.5, lineHeight: 1.5, fontSize: '0.7rem' }}
        >
          {data.summary}
        </Typography>
      )}

      {/* Call stacks — rendered flow diagrams */}
      {callStacks.map((stack, si) => (
        <CallStackView key={si} stack={stack} />
      ))}
    </Box>
  );
}
