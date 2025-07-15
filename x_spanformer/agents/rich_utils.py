import time
from collections import Counter
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def rich_log(
    extracted_data: dict,
    title: str = "Extracted Details",
    color: str = "cyan",
):
    """Logs extracted data in a rich-formatted panel."""
    panel_content = Text()
    for key, value in extracted_data.items():
        panel_content.append(f"{key.capitalize()}: ", style="bold")
        panel_content.append(str(value), style="none")
        panel_content.append("\n")

    console.print(
        Panel(
            panel_content,
            title=title,
            border_style=color,
            expand=False,
        )
    )


def display_telemetry_panel(
    processed_count: int,
    total_count: int,
    start_time: float,
    save_count: int,
    estimated_total_saves: int,
    records_saved_this_session: int,
    keep_count: int | None = None,
    session_start_count: int = 0,  # New parameter for records already processed at session start
):
    """Display a comprehensive telemetry panel in magenta."""
    current_time = time.time()
    elapsed_time = current_time - start_time
    elapsed_minutes = elapsed_time / 60

    # Calculate progress metrics based on actual processing this session
    records_processed_this_session = processed_count - session_start_count
    progress_percentage = (
        (processed_count / total_count * 100) if total_count > 0 else 0
    )
    processing_rate = (
        records_processed_this_session / elapsed_minutes if elapsed_minutes > 0 else 0
    )

    # Estimate remaining time based on actual processing rate
    remaining_items = total_count - processed_count
    estimated_remaining_minutes = (
        (remaining_items / processing_rate) if processing_rate > 0 else 0
    )
    estimated_completion = datetime.now() + timedelta(
        minutes=estimated_remaining_minutes
    )

    # Format time strings
    elapsed_str = f"{int(elapsed_minutes)}m {int(elapsed_time % 60)}s"
    remaining_str = f"{int(estimated_remaining_minutes)}m {int((estimated_remaining_minutes % 1) * 60)}s"
    completion_str = estimated_completion.strftime("%H:%M:%S")

    # Create the telemetry table
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold white", width=20)
    table.add_column("Value", style="bright_cyan", width=25)
    table.add_column("Extra", style="dim white", width=30)

    table.add_row(
        "📊 Progress",
        f"{processed_count:,} / {total_count:,}",
        f"({progress_percentage:.1f}%)",
    )
    
    # Show session-specific processing info
    if session_start_count > 0:
        table.add_row(
            "🚀 This Session",
            f"{records_processed_this_session:,} processed",
            f"({session_start_count:,} resumed)",
        )
    
    # Show breakdown of keeps vs discards if available
    if keep_count is not None:
        discard_count = processed_count - keep_count
        table.add_row(
            "✅ Keep/❌ Discard",
            f"{keep_count:,} / {discard_count:,}",
            f"({keep_count/processed_count*100:.1f}% / {discard_count/processed_count*100:.1f}%)" if processed_count > 0 else "",
        )
    table.add_row(
        "⚡ Processing Rate",
        f"{processing_rate:.1f} segments/min",
        f"{processing_rate * 60:.1f} segments/hour",
    )
    table.add_row(
        "⏱️  Elapsed Time",
        elapsed_str,
        f"Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}",
    )
    table.add_row(
        "🔮 Time Remaining", remaining_str, f"ETA: {completion_str}"
    )
    table.add_row(
        "💾 Save Progress",
        f"{save_count:,} / ~{estimated_total_saves:,}",
        f"({save_count / estimated_total_saves * 100:.1f}%)"
        if estimated_total_saves > 0
        else "",
    )
    table.add_row(
        "📝 Records Saved", f"{records_saved_this_session:,}", "this session"
    )

    # Create the panel
    panel = Panel(
        table,
        title="🚀 [bold white]PROCESSING TELEMETRY[/bold white] 🚀",
        border_style="bright_magenta",
        expand=False,
        width=80,  # Fixed width for consistent box sizing
    )

    # Display with extra spacing
    console.print()
    console.print(panel)
    console.print()


def display_summary_panel(name: str, stats: Counter, reasons: list[str]):
    """Displays a summary panel for processing stats."""
    table = Table(
        title=f"Summary for {name}", show_header=False, box=None, padding=0
    )
    table.add_column("Status", style="bold", width=15)
    table.add_column("Count", style="white")

    for k in ("keep", "discard"):
        count = stats.get(k, 0)
        color = {"keep": "green", "discard": "red"}[k]
        table.add_row(f"[{color}]{k.capitalize()}", f"{count:,}")

    top_reasons = Counter(reasons).most_common(5)
    if top_reasons:
        reasons_text = Text("\nTop Reasons", style="bold blue")
        for r, n in top_reasons:
            reasons_text.append(f"\n• {r} ", style="white")
            reasons_text.append(f"({n}x)", style="dim")
    else:
        reasons_text = Text("")

    content_table = Table.grid(expand=True)
    content_table.add_row(table)
    if top_reasons:
        content_table.add_row(reasons_text)

    panel = Panel(
        content_table,
        title="📊 [bold white]PROCESSING SUMMARY[/bold white] 📊",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)


def display_judgment_result(
    idx: int,
    text: str,
    status: str,
    score: float,
    reason: str,
    content_type: str = "natural",
    total_count: int | None = None
):
    """Display individual judgment result in a styled panel."""
    # Truncate text for display
    display_text = text[:80] + "..." if len(text) > 80 else text
    
    # Choose colors and icons based on status
    if status == "keep":
        border_color = "green"
        status_icon = "✅"
        status_style = "bold green"
    else:
        border_color = "red"
        status_icon = "❌"
        status_style = "bold red"
    
    # Create content table
    table = Table(show_header=False, box=None, padding=0, expand=True)
    table.add_column("Field", style="bold", width=15)
    table.add_column("Value", style="white")
    
    progress_info = f"({idx + 1}/{total_count})" if total_count else f"#{idx + 1}"
    table.add_row("📊 Progress", progress_info)
    
    # Truncate text to ensure consistent width
    text_display = f'"{display_text}"'
    if len(text_display) > 80:
        text_display = text_display[:77] + '..."'
    table.add_row("📝 Text", text_display)
    
    table.add_row("⚖️ Score", f"{score:.2f}")
    table.add_row("🏷️ Type", content_type)
    table.add_row(f"{status_icon} Status", f"[{status_style}]{status.upper()}[/{status_style}]")
    
    # Truncate reason to ensure consistent width
    reason_display = reason[:80] + "..." if len(reason) > 80 else reason
    table.add_row("💭 Reason", reason_display)
    
    panel = Panel(
        table,
        title=f"⚖️ [bold white]JUDGMENT COMPLETE[/bold white] ⚖️",
        border_style=border_color,
        expand=False,
        width=110,  # Fixed width for consistent box sizing
    )
    
    console.print(panel)
    console.print()  # Add spacing after each judgment
