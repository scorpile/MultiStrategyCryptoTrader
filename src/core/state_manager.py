"""SQLite-backed state management utilities for the simulated trading bot."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class StateManager:
    """Coordinates every read/write against the local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        """Store the path to the SQLite file; call `init_db()` before use."""
        self.db_path = db_path

    # ------------------------------------------------------------------ #
    # DB bootstrap helpers
    # ------------------------------------------------------------------ #
    def init_db(self) -> None:
        """Create the SQLite database and all Phase 02 tables if missing."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_parameters (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simulated_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    position_side TEXT,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            try:
                conn.execute("ALTER TABLE simulated_trades ADD COLUMN position_side TEXT;")
            except Exception:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_date TEXT NOT NULL UNIQUE,
                    pnl REAL,
                    trades_count INTEGER,
                    win_rate REAL,
                    drawdown REAL,
                    sharpe REAL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Add new columns if they don't exist
            try:
                conn.execute("ALTER TABLE daily_metrics ADD COLUMN expectancy REAL DEFAULT 0;")
            except:
                pass
            try:
                conn.execute("ALTER TABLE daily_metrics ADD COLUMN profit_factor REAL DEFAULT 0;")
            except:
                pass
            try:
                conn.execute("ALTER TABLE daily_metrics ADD COLUMN avg_win REAL DEFAULT 0;")
            except:
                pass
            try:
                conn.execute("ALTER TABLE daily_metrics ADD COLUMN avg_loss REAL DEFAULT 0;")
            except:
                pass
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS probability_flags (
                    flag_name TEXT PRIMARY KEY,
                    is_enabled INTEGER NOT NULL,
                    probability REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    decision TEXT NOT NULL,
                    price REAL,
                    suggested_size REAL,
                    confidence REAL,
                    executed INTEGER NOT NULL,
                    blocked_reason TEXT,
                    trade_time TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_time TEXT NOT NULL,
                    symbol TEXT,
                    total_value REAL NOT NULL,
                    cash_balance REAL,
                    realized_pnl REAL,
                    open_positions_count INTEGER,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_time TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    order_id TEXT,
                    client_order_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_positions (
                    symbol TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    cost_basis_usdt REAL NOT NULL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT,
                    PRIMARY KEY (symbol, mode)
                );
                """
            )
            try:
                conn.execute("ALTER TABLE execution_positions ADD COLUMN cost_basis_usdt REAL NOT NULL DEFAULT 0;")
            except Exception:
                pass

    def reset_db(self) -> None:
        """Wipe all persisted runtime data (keeps schema)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM simulated_trades;")
            conn.execute("DELETE FROM daily_metrics;")
            conn.execute("DELETE FROM probability_flags;")
            conn.execute("DELETE FROM decision_events;")
            conn.execute("DELETE FROM equity_snapshots;")
            conn.execute("DELETE FROM execution_trades;")
            conn.execute("DELETE FROM execution_positions;")
            conn.execute("DELETE FROM system_parameters;")

    def wipe_db_file(self) -> None:
        """Best-effort: delete the SQLite file (and WAL/SHM) to guarantee a fresh start."""
        try:
            if self.db_path.exists():
                self.db_path.unlink()
        except Exception:
            pass
        for suffix in ("-wal", "-shm"):
            try:
                extra = self.db_path.with_name(self.db_path.name + suffix)
                if extra.exists():
                    extra.unlink()
            except Exception:
                pass

    def ensure_execution_tables(self) -> None:
        """Best-effort: ensure live execution tables exist."""
        self.init_db()

    def count_execution_positions(self, *, mode: Optional[str] = None, symbol: Optional[str] = None) -> int:
        """Return number of rows in execution_positions (optionally filtered)."""
        clauses: list[str] = []
        params: list[Any] = []
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM execution_positions {where};", params).fetchone()
        return int(row[0] if row else 0)

    # ------------------------------------------------------------------ #
    # Parameters
    # ------------------------------------------------------------------ #
    def save_parameters(self, params: Dict[str, Any]) -> None:
        """Persist the provided parameter dictionary using an UPSERT."""
        timestamp = self._utcnow()
        records = [
            (key, json.dumps(value), timestamp)
            for key, value in params.items()
        ]
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO system_parameters (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at;
                """,
                records,
            )

    def load_parameters(self) -> Dict[str, Any]:
        """Return all stored parameters as a dictionary."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT key, value FROM system_parameters;").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # ------------------------------------------------------------------ #
    # Trades
    # ------------------------------------------------------------------ #
    def save_simulated_trade(self, trade_payload: Dict[str, Any]) -> None:
        """Insert a paper trade (order simulation) into the ledger table."""
        normalized = {
            "trade_time": trade_payload.get("trade_time", self._utcnow()),
            "symbol": trade_payload["symbol"],
            "side": trade_payload["side"],
            "position_side": (str(trade_payload.get("position_side") or trade_payload.get("side") or "long")).lower(),
            "quantity": float(trade_payload["quantity"]),
            "price": float(trade_payload["price"]),
            "fee": float(trade_payload.get("fee", 0.0)),
            "pnl": float(trade_payload.get("pnl", 0.0)),
            "metadata": json.dumps(trade_payload.get("metadata", {})),
            "created_at": self._utcnow(),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO simulated_trades (
                    trade_time, symbol, side, position_side, quantity, price, fee, pnl, metadata, created_at
                ) VALUES (
                    :trade_time, :symbol, :side, :position_side, :quantity, :price, :fee, :pnl, :metadata, :created_at
                );
                """,
                normalized,
            )

    def get_trades_for_day(self, date_iso: str, *, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return trades whose timestamp matches YYYY-MM-DD (UTC), optionally filtered by symbol."""
        pattern = f"{date_iso}%"
        clauses = ["trade_time LIKE ?"]
        params: list[Any] = [pattern]
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).upper())
        where = " AND ".join(clauses)
        query = f"SELECT * FROM simulated_trades WHERE {where} ORDER BY trade_time ASC;"
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def migrate_simulated_pnl_net_v1(self) -> bool:
        """One-time migration: recompute SELL `pnl` as net of entry+exit fees.

        Older versions stored SELL pnl as gross (fees excluded). This migration
        replays the ledger using an average-cost lot model and overwrites SELL
        rows with net pnl, improving comparability with live execution.
        """
        try:
            params = self.load_parameters()
            if params.get("simulated_pnl_net_v1") is True:
                return False
        except Exception:
            params = {}

        try:
            trades = self.get_all_trades()
        except Exception:
            trades = []
        if not trades:
            try:
                self.save_parameters({"simulated_pnl_net_v1": True, "simulated_pnl_net_v1_at": self._utcnow()})
            except Exception:
                pass
            return False

        positions: Dict[str, Dict[str, float]] = {}
        updates: list[tuple[float, int]] = []

        def _f(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        for trade in trades:
            try:
                trade_id = int(trade.get("id"))
            except Exception:
                continue
            sym = str(trade.get("symbol", "")).upper()
            side = str(trade.get("side", "")).upper()
            qty = _f(trade.get("quantity"))
            price = _f(trade.get("price"))
            fee = _f(trade.get("fee"))
            if not sym or qty <= 0 or price <= 0 or side not in {"BUY", "SELL"}:
                continue

            pos = positions.setdefault(sym, {"qty": 0.0, "avg_price": 0.0, "cost_basis": 0.0})
            pos_qty = float(pos.get("qty", 0.0) or 0.0)
            pos_avg = float(pos.get("avg_price", 0.0) or 0.0)
            pos_cost = float(pos.get("cost_basis", 0.0) or 0.0)

            if side == "BUY":
                cost = qty * price + fee
                new_qty = pos_qty + qty
                if new_qty > 0:
                    new_avg = ((pos_avg * pos_qty) + (price * qty)) / new_qty if pos_qty > 0 else price
                else:
                    new_avg = 0.0
                pos["qty"] = float(new_qty)
                pos["avg_price"] = float(new_avg)
                pos["cost_basis"] = float(pos_cost + cost)
                continue

            # SELL
            if pos_qty <= 0:
                continue
            sell_qty = min(qty, pos_qty)
            ratio = sell_qty / qty if qty > 0 else 1.0
            proceeds = sell_qty * price - (fee * ratio)
            entry_cost = pos_cost * (sell_qty / pos_qty) if pos_qty > 0 else 0.0
            pnl_net = proceeds - entry_cost

            # Update position state
            pos["qty"] = float(max(0.0, pos_qty - sell_qty))
            pos["cost_basis"] = float(max(0.0, pos_cost - entry_cost))
            if pos["qty"] <= 1e-12:
                pos["qty"] = 0.0
                pos["avg_price"] = 0.0
                pos["cost_basis"] = 0.0

            updates.append((float(pnl_net), trade_id))

        if updates:
            with self._get_connection() as conn:
                conn.executemany(
                    "UPDATE simulated_trades SET pnl = ? WHERE id = ?;",
                    updates,
                )

        try:
            self.save_parameters({"simulated_pnl_net_v1": True, "simulated_pnl_net_v1_at": self._utcnow()})
        except Exception:
            pass
        return bool(updates)

    # ------------------------------------------------------------------ #
    # Decisions / signals (executed + blocked)
    # ------------------------------------------------------------------ #
    def save_decision_event(self, event: Dict[str, Any]) -> None:
        """Persist a decision event (executed trade or blocked signal)."""
        normalized = {
            "event_time": event.get("event_time", self._utcnow()),
            "symbol": str(event.get("symbol", "")).upper(),
            "strategy": event.get("strategy"),
            "decision": str(event.get("decision", "")).lower(),
            "price": float(event.get("price", 0.0) or 0.0) if event.get("price") is not None else None,
            "suggested_size": float(event.get("suggested_size", 0.0) or 0.0) if event.get("suggested_size") is not None else None,
            "confidence": float(event.get("confidence", 0.0) or 0.0) if event.get("confidence") is not None else None,
            "executed": 1 if bool(event.get("executed", False)) else 0,
            "blocked_reason": event.get("blocked_reason"),
            "trade_time": event.get("trade_time"),
            "metadata": json.dumps(event.get("metadata", {})),
            "created_at": self._utcnow(),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_events (
                    event_time, symbol, strategy, decision, price, suggested_size, confidence,
                    executed, blocked_reason, trade_time, metadata, created_at
                ) VALUES (
                    :event_time, :symbol, :strategy, :decision, :price, :suggested_size, :confidence,
                    :executed, :blocked_reason, :trade_time, :metadata, :created_at
                );
                """,
                normalized,
            )

    def count_decisions_since(
        self,
        since: datetime,
        *,
        symbol: Optional[str] = None,
        executed: Optional[bool] = None,
    ) -> int:
        """Count decision events since a given UTC datetime."""
        since_iso = since.strftime(ISO_FORMAT)
        clauses = ["event_time >= ?"]
        params: list[Any] = [since_iso]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if executed is not None:
            clauses.append("executed = ?")
            params.append(1 if executed else 0)
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM decision_events WHERE {where};", params).fetchone()
        return int(row[0] if row else 0)

    def get_decision_stats_for_day(self, date_iso: str, *, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregated decision stats for a given YYYY-MM-DD."""
        pattern = f"{date_iso}%"
        clauses = ["event_time LIKE ?"]
        params: list[Any] = [pattern]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT executed, blocked_reason, COUNT(*) AS cnt
                FROM decision_events
                WHERE {where}
                GROUP BY executed, blocked_reason;
                """,
                params,
            ).fetchall()
        summary: Dict[str, Any] = {"total": 0, "executed": 0, "blocked": 0, "blocked_by_reason": {}}
        for executed_flag, reason, cnt in rows:
            cnt_i = int(cnt or 0)
            summary["total"] += cnt_i
            if int(executed_flag or 0) == 1:
                summary["executed"] += cnt_i
            else:
                summary["blocked"] += cnt_i
                reason_key = str(reason or "unknown")
                summary["blocked_by_reason"][reason_key] = int(summary["blocked_by_reason"].get(reason_key, 0)) + cnt_i
        return summary

    def get_decision_stats_since(self, since: datetime, *, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregated decision stats since a given UTC datetime."""
        since_iso = since.strftime(ISO_FORMAT)
        clauses = ["event_time >= ?"]
        params: list[Any] = [since_iso]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT executed, blocked_reason, COUNT(*) AS cnt
                FROM decision_events
                WHERE {where}
                GROUP BY executed, blocked_reason;
                """,
                params,
            ).fetchall()
        summary: Dict[str, Any] = {"total": 0, "executed": 0, "blocked": 0, "blocked_by_reason": {}}
        for executed_flag, reason, cnt in rows:
            cnt_i = int(cnt or 0)
            summary["total"] += cnt_i
            if int(executed_flag or 0) == 1:
                summary["executed"] += cnt_i
            else:
                summary["blocked"] += cnt_i
                reason_key = str(reason or "unknown")
                summary["blocked_by_reason"][reason_key] = int(summary["blocked_by_reason"].get(reason_key, 0)) + cnt_i
        return summary

    def get_last_decision_event_id(self) -> int:
        """Return the latest decision_events.id or 0 if none."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT MAX(id) FROM decision_events;").fetchone()
        return int(row[0] if row and row[0] is not None else 0)

    def count_total_trades(self, *, symbol: Optional[str] = None) -> int:
        """Return total simulated trade count (optionally filtered by symbol)."""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM simulated_trades {where};", params).fetchone()
        return int(row[0] if row else 0)

    def count_trades_since(
        self,
        since: datetime,
        *,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
    ) -> int:
        """Return the number of simulated trades executed since the given UTC datetime."""
        since_iso = since.strftime(ISO_FORMAT)
        clauses = ["trade_time >= ?"]
        params: list[Any] = [since_iso]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if side:
            clauses.append("side = ?")
            params.append(side.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM simulated_trades WHERE {where};", params).fetchone()
        return int(row[0] if row else 0)

    def get_last_trade_time(
        self,
        *,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
    ) -> Optional[datetime]:
        """Return the most recent trade timestamp (UTC) matching filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if side:
            clauses.append("side = ?")
            params.append(side.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            row = conn.execute(
                f"SELECT trade_time FROM simulated_trades {where} ORDER BY trade_time DESC LIMIT 1;",
                params,
            ).fetchone()
        if not row:
            return None
        return self._parse_iso_z(row[0])

    def get_recent_trades(
        self,
        *,
        limit: int = 200,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent trades, newest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if side:
            clauses.append("side = ?")
            params.append(side.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM simulated_trades {where} ORDER BY id DESC LIMIT ?;",
                (*params, int(limit)),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Execution trades/positions (live or paper routing)
    # ------------------------------------------------------------------ #
    def save_execution_trade(self, trade: Dict[str, Any], *, mode: str) -> None:
        payload = {
            "trade_time": trade.get("trade_time") or self._utcnow(),
            "mode": str(mode or "live"),
            "symbol": str(trade.get("symbol") or "").upper(),
            "side": str(trade.get("side") or "").upper(),
            "quantity": float(trade.get("quantity", 0.0) or 0.0),
            "price": float(trade.get("price", 0.0) or 0.0),
            "fee": float(trade.get("fee", 0.0) or 0.0),
            "pnl": float(trade.get("pnl", 0.0) or 0.0),
            "order_id": str(trade.get("order_id") or "") or None,
            "client_order_id": str(trade.get("client_order_id") or "") or None,
            "metadata": json.dumps(trade.get("metadata", {})),
            "created_at": self._utcnow(),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO execution_trades (
                    trade_time, mode, symbol, side, quantity, price, fee, pnl, order_id, client_order_id, metadata, created_at
                ) VALUES (
                    :trade_time, :mode, :symbol, :side, :quantity, :price, :fee, :pnl, :order_id, :client_order_id, :metadata, :created_at
                );
                """,
                payload,
            )

    def count_execution_orders_for_day(self, date_iso: str, *, mode: str) -> int:
        pattern = f"{date_iso}%"
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM execution_trades
                WHERE trade_time LIKE ? AND mode = ?;
                """,
                (pattern, str(mode or "live")),
            ).fetchone()
        return int(row[0] if row else 0)

    def get_execution_realized_pnl_for_day(self, date_iso: str, *, mode: str) -> float:
        pattern = f"{date_iso}%"
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM execution_trades
                WHERE trade_time LIKE ? AND mode = ?;
                """,
                (pattern, str(mode or "live")),
            ).fetchone()
        try:
            return float(row[0] if row else 0.0)
        except Exception:
            return 0.0

    def get_execution_realized_pnl_total(self, *, mode: str) -> float:
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) FROM execution_trades
                WHERE mode = ?;
                """,
                (str(mode or "live"),),
            ).fetchone()
        try:
            return float(row[0] if row else 0.0)
        except Exception:
            return 0.0

    def get_execution_position(self, *, symbol: str, mode: str) -> Optional[Dict[str, Any]]:
        sym = str(symbol or "").upper()
        if not sym:
            return None
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT * FROM execution_positions WHERE symbol = ? AND mode = ?;
                """,
                (sym, str(mode or "live")),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def upsert_execution_position(
        self,
        *,
        symbol: str,
        mode: str,
        quantity_delta: float,
        execution_price: float,
        cost_basis_delta_usdt: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sym = str(symbol or "").upper()
        if not sym:
            raise ValueError("symbol required")
        qty_delta = float(quantity_delta or 0.0)
        px = float(execution_price or 0.0)
        if qty_delta <= 0 or px <= 0:
            raise ValueError("quantity_delta and execution_price must be > 0")

        existing = self.get_execution_position(symbol=sym, mode=mode) or {}
        old_qty = float(existing.get("quantity", 0.0) or 0.0)
        old_avg = float(existing.get("avg_price", 0.0) or 0.0)
        old_cost = float(existing.get("cost_basis_usdt", 0.0) or 0.0)
        new_qty = old_qty + qty_delta
        new_avg = px if old_qty <= 0 or old_avg <= 0 else ((old_qty * old_avg + qty_delta * px) / new_qty)
        delta_cost = float(cost_basis_delta_usdt) if cost_basis_delta_usdt is not None else (qty_delta * px)
        new_cost = old_cost + max(0.0, float(delta_cost))

        payload = {
            "symbol": sym,
            "mode": str(mode or "live"),
            "quantity": float(new_qty),
            "avg_price": float(new_avg),
            "cost_basis_usdt": float(new_cost),
            "stop_loss": float(stop_loss) if stop_loss is not None else existing.get("stop_loss"),
            "take_profit": float(take_profit) if take_profit is not None else existing.get("take_profit"),
            "updated_at": self._utcnow(),
            "metadata": json.dumps(metadata or (existing.get("metadata") or {})),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO execution_positions (
                    symbol, mode, quantity, avg_price, cost_basis_usdt, stop_loss, take_profit, updated_at, metadata
                ) VALUES (
                    :symbol, :mode, :quantity, :avg_price, :cost_basis_usdt, :stop_loss, :take_profit, :updated_at, :metadata
                )
                ON CONFLICT(symbol, mode) DO UPDATE SET
                    quantity = excluded.quantity,
                    avg_price = excluded.avg_price,
                    cost_basis_usdt = excluded.cost_basis_usdt,
                    stop_loss = excluded.stop_loss,
                    take_profit = excluded.take_profit,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata;
                """,
                payload,
            )
        return payload

    def reduce_execution_position_and_realize_pnl(
        self,
        *,
        symbol: str,
        mode: str,
        sell_quantity: float,
        sell_price: float,
        sell_fee_usdt: float = 0.0,
    ) -> float:
        sym = str(symbol or "").upper()
        if not sym:
            raise ValueError("symbol required")
        qty = float(sell_quantity or 0.0)
        px = float(sell_price or 0.0)
        if qty <= 0 or px <= 0:
            raise ValueError("sell_quantity and sell_price must be > 0")

        pos = self.get_execution_position(symbol=sym, mode=mode) or {}
        old_qty = float(pos.get("quantity", 0.0) or 0.0)
        old_avg = float(pos.get("avg_price", 0.0) or 0.0)
        old_cost = float(pos.get("cost_basis_usdt", 0.0) or 0.0)
        if old_qty <= 0 or old_avg <= 0:
            raise ValueError("No open position to reduce.")

        realized_qty = min(qty, old_qty)
        fee_scaled = float(sell_fee_usdt or 0.0) * (realized_qty / qty) if qty > 0 else float(sell_fee_usdt or 0.0)
        proceeds = realized_qty * px - fee_scaled
        entry_cost = old_cost * (realized_qty / old_qty) if old_qty > 0 else 0.0
        pnl = proceeds - entry_cost
        new_qty = old_qty - realized_qty
        new_cost = max(0.0, old_cost - entry_cost)

        with self._get_connection() as conn:
            if new_qty <= 1e-12:
                conn.execute(
                    "DELETE FROM execution_positions WHERE symbol = ? AND mode = ?;",
                    (sym, str(mode or "live")),
                )
            else:
                conn.execute(
                    """
                    UPDATE execution_positions
                    SET quantity = ?, cost_basis_usdt = ?, updated_at = ?
                    WHERE symbol = ? AND mode = ?;
                    """,
                    (float(new_qty), float(new_cost), self._utcnow(), sym, str(mode or "live")),
                )
        return float(pnl)

    def get_recent_round_trips(
        self,
        *,
        limit: int = 20,
        symbol: Optional[str] = None,
        max_lookback_trades: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Return recent closed 'round trips' including entry/exit costs, triggers, and net PnL.

        Notes:
        - Uses FIFO lots per symbol.
        - Supports both long (BUY then SELL) and short (SELL then BUY) flows.
        - Net PnL accounts for both entry + exit fees.
        """
        lookback = max(int(max_lookback_trades), int(limit) * 20, 200)
        trades = self.get_recent_trades(limit=lookback, symbol=symbol)
        if not trades:
            return []

        trades_sorted = list(reversed(trades))  # oldest -> newest
        long_lots: Dict[str, List[Dict[str, Any]]] = {}
        short_lots: Dict[str, List[Dict[str, Any]]] = {}
        trips: List[Dict[str, Any]] = []

        def _f(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        def _parse_meta(meta_raw: Any) -> tuple[Dict[str, Any], str, str]:
            meta = meta_raw or {}
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except Exception:
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}
            decision_meta = meta.get("decision_metadata") or {}
            trigger = meta.get("trigger") or decision_meta.get("trigger") or ""
            reason = meta.get("reason") or decision_meta.get("reason") or ""
            if not reason:
                try:
                    reasons = meta.get("reasons") or decision_meta.get("reasons") or []
                    if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
                        first = next(iter(reasons), "")
                        reason = str(first or "")
                    elif reasons:
                        reason = str(reasons)
                except Exception:
                    reason = ""
            return meta, str(trigger or ""), str(reason or "")

        for trade in trades_sorted:
            sym = str(trade.get("symbol", "")).upper()
            side = str(trade.get("side", "")).upper()
            qty = _f(trade.get("quantity"))
            price = _f(trade.get("price"))
            fee = _f(trade.get("fee"))
            ts = trade.get("trade_time") or trade.get("timestamp") or trade.get("created_at") or ""
            meta, trigger, reason = _parse_meta(trade.get("metadata"))

            if qty <= 0 or price <= 0 or side not in {"BUY", "SELL"}:
                continue

            if side == "SELL":
                # Close longs first; remainder opens/extends a short.
                remaining = qty
                lots = long_lots.setdefault(sym, [])
                entry_qty = 0.0
                entry_notional = 0.0
                entry_fee = 0.0
                entry_time = None
                entry_prices: list[float] = []
                entry_triggers: list[str] = []
                entry_reasons: list[str] = []

                while remaining > 1e-12 and lots:
                    lot = lots[0]
                    if lot.get("symbol") != sym:
                        lots.pop(0)
                        continue
                    lot_qty = float(lot.get("qty", 0.0) or 0.0)
                    if lot_qty <= 1e-12:
                        lots.pop(0)
                        continue
                    use_qty = min(lot_qty, remaining)
                    if entry_time is None:
                        entry_time = lot.get("time")

                    entry_qty += use_qty
                    lot_price = float(lot.get("price", 0.0) or 0.0)
                    entry_notional += use_qty * lot_price
                    entry_prices.append(lot_price)
                    if lot.get("trigger"):
                        entry_triggers.append(str(lot.get("trigger")))
                    if lot.get("reason"):
                        entry_reasons.append(str(lot.get("reason")))

                    lot_fee_rem = float(lot.get("fee", 0.0) or 0.0)
                    alloc_fee = (lot_fee_rem * (use_qty / lot_qty)) if lot_qty > 0 else 0.0
                    entry_fee += alloc_fee

                    lot["qty"] = lot_qty - use_qty
                    lot["fee"] = lot_fee_rem - alloc_fee
                    remaining -= use_qty
                    if float(lot.get("qty", 0.0) or 0.0) <= 1e-12:
                        lots.pop(0)

                if entry_qty > 0:
                    entry_avg_price = entry_notional / max(entry_qty, 1e-12)
                    exit_fee = fee * (entry_qty / qty) if qty > 0 else fee
                    entry_cost = (entry_avg_price * entry_qty) + entry_fee
                    exit_value = (price * entry_qty) - exit_fee
                    gross_pnl = (price - entry_avg_price) * entry_qty
                    total_fees = entry_fee + exit_fee
                    net_pnl = gross_pnl - total_fees

                    trips.append(
                        {
                            "symbol": sym,
                            "quantity": round(entry_qty, 6),
                            "direction": "long",
                            "entry_time": entry_time or "",
                            "exit_time": ts,
                            "entry_price": entry_avg_price,
                            "exit_price": price,
                            "entry_price_min": min(entry_prices) if entry_prices else entry_avg_price,
                            "entry_price_max": max(entry_prices) if entry_prices else entry_avg_price,
                            "exit_price_min": price,
                            "exit_price_max": price,
                            "entry_cost": entry_cost,
                            "exit_value": exit_value,
                            "gross_pnl": gross_pnl,
                            "fees": total_fees,
                            "net_pnl": net_pnl,
                            "trigger": trigger,
                            "entry_trigger": entry_triggers[0] if entry_triggers else "",
                            "exit_trigger": trigger,
                            "entry_reason": entry_reasons[0] if entry_reasons else (entry_triggers[0] if entry_triggers else ""),
                            "exit_reason": reason or trigger,
                        }
                    )

                # Any remaining opens/extends a short position.
                if remaining > 1e-12:
                    short_lots.setdefault(sym, []).append(
                        {
                            "qty": remaining,
                            "price": price,
                            "fee": fee * (remaining / qty) if qty > 0 else fee,
                            "time": ts,
                            "symbol": sym,
                            "trigger": trigger,
                            "reason": reason or trigger,
                        }
                    )
                continue

            # side == BUY
            remaining = qty
            lots = short_lots.setdefault(sym, [])
            entry_qty = 0.0
            entry_notional = 0.0
            entry_fee = 0.0
            entry_time = None
            entry_prices: list[float] = []
            entry_triggers: list[str] = []
            entry_reasons: list[str] = []

            while remaining > 1e-12 and lots:
                lot = lots[0]
                if lot.get("symbol") != sym:
                    lots.pop(0)
                    continue
                lot_qty = float(lot.get("qty", 0.0) or 0.0)
                if lot_qty <= 1e-12:
                    lots.pop(0)
                    continue
                use_qty = min(lot_qty, remaining)
                if entry_time is None:
                    entry_time = lot.get("time")

                entry_qty += use_qty
                lot_price = float(lot.get("price", 0.0) or 0.0)
                entry_notional += use_qty * lot_price
                entry_prices.append(lot_price)
                if lot.get("trigger"):
                    entry_triggers.append(str(lot.get("trigger")))
                if lot.get("reason"):
                    entry_reasons.append(str(lot.get("reason")))

                lot_fee_rem = float(lot.get("fee", 0.0) or 0.0)
                alloc_fee = (lot_fee_rem * (use_qty / lot_qty)) if lot_qty > 0 else 0.0
                entry_fee += alloc_fee

                lot["qty"] = lot_qty - use_qty
                lot["fee"] = lot_fee_rem - alloc_fee
                remaining -= use_qty
                if float(lot.get("qty", 0.0) or 0.0) <= 1e-12:
                    lots.pop(0)

            if entry_qty > 0:
                entry_avg_price = entry_notional / max(entry_qty, 1e-12)
                exit_fee = fee * (entry_qty / qty) if qty > 0 else fee
                entry_cost = (entry_avg_price * entry_qty) - entry_fee  # short entry = proceeds after fee
                exit_value = (price * entry_qty) + exit_fee  # cash out to close short
                gross_pnl = (entry_avg_price - price) * entry_qty
                total_fees = entry_fee + exit_fee
                net_pnl = gross_pnl - total_fees

                trips.append(
                    {
                        "symbol": sym,
                        "quantity": round(entry_qty, 6),
                        "direction": "short",
                        "entry_time": entry_time or "",
                        "exit_time": ts,
                        "entry_price": entry_avg_price,
                        "exit_price": price,
                        "entry_price_min": min(entry_prices) if entry_prices else entry_avg_price,
                        "entry_price_max": max(entry_prices) if entry_prices else entry_avg_price,
                        "exit_price_min": price,
                        "exit_price_max": price,
                        "entry_cost": entry_cost,
                        "exit_value": exit_value,
                        "gross_pnl": gross_pnl,
                        "fees": total_fees,
                        "net_pnl": net_pnl,
                        "trigger": trigger,
                        "entry_trigger": entry_triggers[0] if entry_triggers else "",
                        "exit_trigger": trigger,
                        "entry_reason": entry_reasons[0] if entry_reasons else (entry_triggers[0] if entry_triggers else ""),
                        "exit_reason": reason or trigger,
                    }
                )

            # Any remaining opens/extends a long position.
            if remaining > 1e-12:
                long_lots.setdefault(sym, []).append(
                    {
                        "qty": remaining,
                        "price": price,
                        "fee": fee * (remaining / qty) if qty > 0 else fee,
                        "time": ts,
                        "symbol": sym,
                        "trigger": trigger,
                        "reason": reason or trigger,
                    }
                )

        trips = list(reversed(trips))  # newest first
        return trips[: int(limit)]

    def get_all_trades(
        self,
        *,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all simulated trades ordered oldest->newest (for state rehydration)."""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM simulated_trades {where} ORDER BY id ASC;",
                params,
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Equity snapshots
    # ------------------------------------------------------------------ #
    def save_equity_snapshot(self, snapshot: Dict[str, Any]) -> None:
        payload = {
            "snapshot_time": snapshot.get("snapshot_time", self._utcnow()),
            "symbol": (str(snapshot.get("symbol")).upper() if snapshot.get("symbol") else None),
            "total_value": float(snapshot.get("total_value", 0.0) or 0.0),
            "cash_balance": float(snapshot.get("cash_balance", 0.0) or 0.0) if snapshot.get("cash_balance") is not None else None,
            "realized_pnl": float(snapshot.get("realized_pnl", 0.0) or 0.0) if snapshot.get("realized_pnl") is not None else None,
            "open_positions_count": int(snapshot.get("open_positions_count", 0) or 0) if snapshot.get("open_positions_count") is not None else None,
            "metadata": json.dumps(snapshot.get("metadata", {})),
            "created_at": self._utcnow(),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots (
                    snapshot_time, symbol, total_value, cash_balance, realized_pnl, open_positions_count, metadata, created_at
                ) VALUES (
                    :snapshot_time, :symbol, :total_value, :cash_balance, :realized_pnl, :open_positions_count, :metadata, :created_at
                );
                """,
                payload,
            )

    def get_equity_since(self, since: datetime, *, symbol: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        since_iso = since.strftime(ISO_FORMAT)
        clauses = ["snapshot_time >= ?"]
        params: list[Any] = [since_iso]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT snapshot_time, total_value, cash_balance, realized_pnl, open_positions_count, metadata
                FROM equity_snapshots
                WHERE {where}
                ORDER BY snapshot_time ASC
                LIMIT ?;
                """,
                (*params, int(limit)),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_equity_day_summary(self, date_iso: str, *, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return {start, latest, peak} equity totals for a given YYYY-MM-DD (UTC)."""
        pattern = f"{date_iso}%"
        clauses = ["snapshot_time LIKE ?"]
        params: list[Any] = [pattern]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    (SELECT total_value FROM equity_snapshots WHERE {where} ORDER BY snapshot_time ASC LIMIT 1) AS start_value,
                    (SELECT total_value FROM equity_snapshots WHERE {where} ORDER BY snapshot_time DESC LIMIT 1) AS latest_value,
                    (SELECT MAX(total_value) FROM equity_snapshots WHERE {where}) AS peak_value,
                    (SELECT COUNT(*) FROM equity_snapshots WHERE {where}) AS snapshots
                """,
                params * 4,
            ).fetchone()
        return {
            "date": date_iso,
            "snapshots": int(row[3] if row and row[3] is not None else 0),
            "start_value": float(row[0]) if row and row[0] is not None else None,
            "latest_value": float(row[1]) if row and row[1] is not None else None,
            "peak_value": float(row[2]) if row and row[2] is not None else None,
        }

    def get_equity_range_summary(
        self,
        start_utc: datetime,
        end_utc: datetime,
        *,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {start, latest, peak} equity totals for a UTC datetime range [start, end)."""
        start_iso = start_utc.strftime(ISO_FORMAT)
        end_iso = end_utc.strftime(ISO_FORMAT)
        clauses = ["snapshot_time >= ?", "snapshot_time < ?"]
        params: list[Any] = [start_iso, end_iso]
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        where = " AND ".join(clauses)
        with self._get_connection() as conn:
            row = conn.execute(
                f"""
                SELECT
                    (SELECT total_value FROM equity_snapshots WHERE {where} ORDER BY snapshot_time ASC LIMIT 1) AS start_value,
                    (SELECT total_value FROM equity_snapshots WHERE {where} ORDER BY snapshot_time DESC LIMIT 1) AS latest_value,
                    (SELECT MAX(total_value) FROM equity_snapshots WHERE {where}) AS peak_value,
                    (SELECT COUNT(*) FROM equity_snapshots WHERE {where}) AS snapshots
                """,
                params * 4,
            ).fetchone()
        return {
            "start_utc": start_iso,
            "end_utc": end_iso,
            "snapshots": int(row[3] if row and row[3] is not None else 0),
            "start_value": float(row[0]) if row and row[0] is not None else None,
            "latest_value": float(row[1]) if row and row[1] is not None else None,
            "peak_value": float(row[2]) if row and row[2] is not None else None,
        }

    # ------------------------------------------------------------------ #
    # Daily metrics
    # ------------------------------------------------------------------ #
    def save_daily_metrics(self, metrics: Dict[str, Any]) -> None:
        """Insert or update daily aggregate metrics produced by the evaluator."""
        payload = {
            "metric_date": metrics["metric_date"],
            "pnl": float(metrics.get("pnl", 0.0)),
            "trades_count": int(metrics.get("trades_count", 0)),
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "drawdown": float(metrics.get("drawdown", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "notes": metrics.get("notes"),
            "created_at": self._utcnow(),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO daily_metrics (
                    metric_date, pnl, trades_count, win_rate, drawdown, sharpe, notes, created_at
                ) VALUES (
                    :metric_date, :pnl, :trades_count, :win_rate, :drawdown, :sharpe, :notes, :created_at
                )
                ON CONFLICT(metric_date) DO UPDATE SET
                    pnl = excluded.pnl,
                    trades_count = excluded.trades_count,
                    win_rate = excluded.win_rate,
                    drawdown = excluded.drawdown,
                    sharpe = excluded.sharpe,
                    notes = excluded.notes,
                    created_at = excluded.created_at;
                """,
                payload,
            )

    def get_recent_metrics(self, limit: int = 7) -> List[Dict[str, Any]]:
        """Return the most recent N daily metrics ordered from newest to oldest."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM daily_metrics
                ORDER BY metric_date DESC
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Probability flag
    # ------------------------------------------------------------------ #
    def set_flag(
        self,
        *,
        flag_name: str,
        enabled: bool,
        probability: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist an arbitrary flag in `probability_flags`."""
        name = str(flag_name or "").strip()
        if not name:
            raise ValueError("flag_name is required")
        payload = {
            "flag_name": name,
            "is_enabled": 1 if enabled else 0,
            "probability": float(probability),
            "updated_at": self._utcnow(),
            "metadata": json.dumps(metadata or {}),
        }
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO probability_flags (
                    flag_name, is_enabled, probability, updated_at, metadata
                ) VALUES (
                    :flag_name, :is_enabled, :probability, :updated_at, :metadata
                )
                ON CONFLICT(flag_name) DO UPDATE SET
                    is_enabled = excluded.is_enabled,
                    probability = excluded.probability,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata;
                """,
                payload,
            )

    def get_flag(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Return a flag row from `probability_flags` or None if missing."""
        name = str(flag_name or "").strip()
        if not name:
            return None
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM probability_flags WHERE flag_name = ?;",
                (name,),
            ).fetchone()
        if not row:
            return None
        row_dict = self._row_to_dict(row)
        row_dict["details"] = row_dict.get("metadata", {})
        return row_dict

    def set_probability_flag(
        self,
        enabled: bool,
        probability: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist the ≥30% probability flag plus optional metadata."""
        self.set_flag(flag_name="profit_consistency", enabled=enabled, probability=probability, metadata=metadata)

    def get_probability_flag(self) -> Optional[Dict[str, Any]]:
        """Return the last known probability flag state or `None` if absent."""
        return self.get_flag("profit_consistency")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_connection(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL enabled."""
        conn = sqlite3.connect(self.db_path.as_posix(), detect_types=sqlite3.PARSE_DECLTYPES)
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert sqlite3.Row into a plain dict decoding JSON fields."""
        result = {key: row[key] for key in row.keys()}
        if "metadata" in result and isinstance(result["metadata"], str):
            try:
                result["metadata"] = json.loads(result["metadata"])
            except json.JSONDecodeError:
                pass
        return result

    @staticmethod
    def _utcnow() -> str:
        """Return current UTC time in ISO-8601 format."""
        return datetime.utcnow().strftime(ISO_FORMAT)

    @staticmethod
    def _parse_iso_z(value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
