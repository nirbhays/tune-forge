# TuneForge Blog Post -- Gemini Image Generation Prompts

> Style bible for all images: Modern, clean, minimalist tech aesthetic. Dark theme using charcoal black (#1e1e2e) as the primary background, orange-amber (#ff9100) as the accent color, and white (#ffffff) for text and line art. Flat design / isometric style -- no stock photo feel. Terminal and CLI visual elements where appropriate. Consistent visual language across all seven images.

---

### Image 1: Hero / Cover Image
**Filename:** `tuneforge-blog-hero.png`
**Dimensions:** 1200x630
**Prompt:**
Create a wide-format hero banner image for a technical blog post. Dark charcoal background (#1e1e2e). In the center, three clean, glowing terminal windows are arranged in a slight horizontal cascade (left to right), each showing one short CLI command: "tuneforge finetune", "tuneforge eval", and "tuneforge serve". The terminal windows have a subtle rounded-corner border in orange-amber (#ff9100) and use a monospace font for the command text in white. Behind and above them, fading into the dark background, is a chaotic tangle of crumpled notebook cells, broken code snippets, and red error badges -- all rendered as faint, ghostly, desaturated outlines to represent the old way. A thin orange-amber horizontal line separates the chaos above from the clean terminals below, implying a before/after divide. In the bottom-right corner, the word "TuneForge" is set in a bold, modern sans-serif font in white with a small orange-amber lightning-bolt icon next to it. Flat design, no gradients except for subtle dark-to-darker vignette on edges. No people, no stock photo elements. The overall mood is: clarity replacing chaos.

**Alt Text:** Hero banner showing three clean TuneForge CLI terminal windows labeled finetune, eval, and serve, emerging from a faded background of chaotic notebook code, representing simplicity replacing complexity.

**Placement:** Top of the blog post, directly below the title and subtitle. This is the first thing the reader sees.

---

### Image 2: The Problem
**Filename:** `tuneforge-blog-problem.png`
**Dimensions:** 800x500
**Prompt:**
Create a split-screen comparison illustration on a dark charcoal (#1e1e2e) background. LEFT SIDE (labeled "The Reality" in small white text at the top): A stylized Jupyter notebook interface rendered in flat design. The notebook shows a long vertical scroll of cells -- at least 12-15 visible cells stacked tightly. Some cells have red error banners with "RuntimeError" and "CUDA out of memory" text. Some cells have orange warning triangles. Cell numbers are jumbled and out of order (e.g., [47], [23], [*], [12]). A small cell counter badge reads "47 cells". Scrollbar on the right is tiny, indicating a very long notebook. The overall palette for this side is desaturated grays and reds -- it should feel stressful and cluttered. RIGHT SIDE (labeled "The Alternative" in small white text at the top): A single clean terminal window with three lines of commands in orange-amber (#ff9100) monospace text: "$ tuneforge finetune ...", "$ tuneforge eval ...", "$ tuneforge serve ...". Each line has a small green checkmark to its left. The terminal has a clean dark background slightly lighter than the outer background. A vertical dashed line in muted gray separates the two halves. The contrast should be stark: visual chaos on the left, elegant simplicity on the right. Flat design, isometric-light style. No people.

**Alt Text:** Split-screen comparison showing a cluttered 47-cell Jupyter notebook with errors and warnings on the left versus three clean TuneForge CLI commands with green checkmarks on the right.

**Placement:** In the "The Problem Nobody Talks About" section, after the paragraph describing the messy fine-tuning workflow.

---

### Image 3: Three Commands Pipeline
**Filename:** `tuneforge-blog-pipeline.png`
**Dimensions:** 800x500
**Prompt:**
Create a horizontal pipeline diagram on a dark charcoal (#1e1e2e) background. Three isometric-style rounded rectangles are arranged left to right, connected by thick orange-amber (#ff9100) arrows flowing between them. FIRST BLOCK: Labeled "finetune" in bold white monospace text. Below the label, small icon representations of a JSONL file going in (left side) and adapter weights coming out (right side). A small gear/cog icon in orange-amber sits inside the block. SECOND BLOCK: Labeled "eval" in bold white monospace text. Below the label, small icon representations of a test dataset going in and a metrics table coming out. A small bar-chart icon in orange-amber sits inside the block. THIRD BLOCK: Labeled "serve" in bold white monospace text. Below the label, small icon representations of adapter weights going in and an API endpoint symbol (a small cloud with a plug) coming out. A small rocket icon in orange-amber sits inside the block. Below all three blocks, a thin horizontal timeline bar in muted gray with labels: "Plan (laptop)" under the first block, "Measure (laptop or GPU)" under the second, and "Deploy (anywhere)" under the third. The arrows between blocks should feel like a confident, unidirectional flow. Clean flat design, no shadows, slight isometric perspective on the blocks. The overall composition is wide and horizontally balanced.

**Alt Text:** Horizontal pipeline diagram showing TuneForge's three commands -- finetune, eval, and serve -- as connected steps flowing left to right with inputs and outputs labeled for each stage.

**Placement:** In the "Three Commands. That Is the Whole Idea." section, above or below the code block showing the three commands.

---

### Image 4: Graceful Degradation
**Filename:** `tuneforge-blog-graceful-degradation.png`
**Dimensions:** 800x500
**Prompt:**
Create an illustration on a dark charcoal (#1e1e2e) background showing two devices side by side with a connection between them. LEFT DEVICE: A flat-design illustration of a slim laptop (like a MacBook Air) shown at a slight isometric angle. On its screen, a terminal window displays the text "tuneforge finetune --dry-run" in orange-amber (#ff9100) monospace text, with a structured output table below it showing parameters like "LoRA rank: 16", "Samples: 8,000", "Status: PLAN READY" in white text. A small airplane icon floats above the laptop, suggesting it is being used on a flight. A label below reads "No GPU Required" in white. RIGHT DEVICE: A flat-design illustration of a rack server or powerful workstation with visible GPU card outlines glowing in orange-amber. On its screen, a terminal window displays "tuneforge finetune" (no --dry-run flag) with a progress bar at 67% in orange-amber and "Training... step 4200/6000" in white text. A label below reads "GPU Execution" in white. BETWEEN THE DEVICES: A curved dashed arrow in orange-amber flows from the laptop to the server, labeled "git push" in small white text along the arrow. The visual story is: plan on the left, execute on the right, connected by version control. Both devices share the same dark aesthetic. No people, flat design, clean lines.

**Alt Text:** Illustration showing a laptop running TuneForge in dry-run mode to plan training without a GPU on the left, connected by a git push arrow to a GPU server executing the actual training on the right.

**Placement:** In the "The Graceful Degradation Pattern" section, after the paragraph explaining the two-tier installation and before the concrete scenario about the ML lead on a flight.

---

### Image 5: Dry-Run Output
**Filename:** `tuneforge-blog-dry-run.png`
**Dimensions:** 800x500
**Prompt:**
Create a realistic-looking terminal screenshot on a dark charcoal (#1e1e2e) background. The image shows a single terminal window with a dark background slightly lighter than the outer background (#252535). At the top of the terminal, a title bar reads "TuneForge -- Dry Run" in a muted gray. Below it, the command "$ tuneforge finetune --model mistralai/Mistral-7B-v0.1 --dataset data/train.jsonl --output ./my-adapter --epochs 3 --lr 2e-4 --batch-size 4 --dry-run" is shown in orange-amber (#ff9100) monospace text, word-wrapped neatly. Below the command, a structured Rich-style table is rendered with clean box-drawing characters in muted gray. The table has two columns: "Parameter" and "Value". Rows include: Model = "mistralai/Mistral-7B-v0.1", Dataset = "data/train.jsonl", Samples = "8,000", LoRA Rank = "16", LoRA Alpha = "32", Learning Rate = "2e-4", Epochs = "3", Batch Size = "4", Max Seq Length = "512", Quantization = "4-bit (QLoRA)", Target Modules = "q_proj, v_proj", Est. Memory = "~6.5 GB", GPU Package = "Not installed". At the bottom, a status line in bright orange-amber reads: "DRY RUN COMPLETE -- Training plan validated. Run without --dry-run to execute." All text is in monospace font. The table lines use Unicode box-drawing characters for a polished look. Clean, sharp, no anti-aliasing artifacts. This should look like a real terminal output that someone could screenshot and share.

**Alt Text:** Terminal screenshot showing TuneForge dry-run output with a structured table displaying all training parameters including model, dataset, LoRA configuration, quantization settings, and a validation success message.

**Placement:** In the "What the Output Looks Like" section, as a visual companion to the description of the dry-run output format.

---

### Image 6: Eval Metrics
**Filename:** `tuneforge-blog-eval-metrics.png`
**Dimensions:** 800x500
**Prompt:**
Create a data visualization image on a dark charcoal (#1e1e2e) background. The image has two visual elements arranged vertically. TOP ELEMENT: A terminal-style table (Rich-style with Unicode box-drawing characters in muted gray) titled "TuneForge Evaluation Results" in white bold text. The table has two columns: "Metric" and "Score". Three rows: exact_match = 0.6667 (with a small yellow-orange dot indicator), contains = 1.0000 (with a small green dot indicator), length_ratio = 1.4833 (with a small blue dot indicator). The table text is in white monospace font. BOTTOM ELEMENT: Three horizontal bar charts directly below the table, one for each metric, providing a visual representation of the scores. The exact_match bar fills to about 67% of its track in yellow-orange. The contains bar fills to 100% of its track in green. The length_ratio bar extends to about 148% with a vertical dashed line at the 100% mark (the "ideal" line) labeled "1.0" and the bar overshooting it in blue, visually showing the model over-generates. Each bar has its metric name to the left and its numeric value to the right. The bars sit on a subtly lighter dark background strip. The overall design is clean and information-dense but not cluttered. No decorative elements, pure data visualization. Flat design.

**Alt Text:** Evaluation metrics visualization showing a results table and horizontal bar charts for three TuneForge metrics: exact_match at 66.67 percent, contains at 100 percent, and length_ratio at 1.48 indicating slight over-generation.

**Placement:** In the "The Eval Command: More Than an Afterthought" section, after the explanation of the three metrics and what they reveal.

---

### Image 7: Comparison
**Filename:** `tuneforge-blog-comparison.png`
**Dimensions:** 800x500
**Prompt:**
Create a comparison diagram on a dark charcoal (#1e1e2e) background. The layout is a horizontal spectrum/axis running left to right across the image. The axis is a thick line that transitions from orange-amber (#ff9100) on the left to muted gray on the right. LEFT END is labeled "Simplicity" in white bold text. RIGHT END is labeled "Features" in white bold text. Three tool cards are positioned along the spectrum. LEFT POSITION (near the Simplicity end): A rounded rectangle card for "TuneForge" with orange-amber border and the label in bold white. Inside the card: "3 commands", "Zero config files", "5-sec install", "Dry-run mode" in small white text, each on its own line. A small orange-amber terminal icon. MIDDLE POSITION: A rounded rectangle card for "LLaMA-Factory" with a muted teal border and label in white. Inside: "Web UI", "Many training methods", "Visual interface" in small white text. A small teal monitor/UI icon. RIGHT POSITION (near the Features end): A rounded rectangle card for "Axolotl" with a muted purple border and label in white. Inside: "Dozens of architectures", "RLHF/DPO", "YAML configs", "Research-grade" in small white text. A small purple Swiss Army knife icon. Below the spectrum, a single-line callout in muted gray italic text reads: "Pick the right tool for your workflow." Thin dashed vertical lines connect each card to its position on the spectrum axis. The composition should feel balanced and fair -- this is an honest comparison, not marketing. No tool is shown as "better," just positioned differently on the simplicity-to-features axis. Flat design, clean geometry.

**Alt Text:** Comparison diagram positioning TuneForge, LLaMA-Factory, and Axolotl on a simplicity-to-features spectrum, with TuneForge offering the simplest workflow, LLaMA-Factory providing a visual interface, and Axolotl delivering the most features.

**Placement:** In the "Honest Comparison: TuneForge vs. Axolotl vs. LLaMA-Factory" section, after the introductory paragraph and before or after the detailed comparison text.

---

## Usage Notes

- **Gemini model:** Use Gemini 2.0 Flash (or Imagen 3 via Gemini) for generation. These prompts are written to be self-contained -- paste the full prompt text into the generation input.
- **Iteration:** If the first generation is not right, try appending "Make the text more legible" or "Increase contrast between background and foreground elements" to the prompt.
- **Color calibration:** The hex values in the prompts (#1e1e2e, #ff9100, #ffffff) are guidelines for the AI model. Verify the output colors match your blog theme and adjust the prompt wording if needed (e.g., "use a slightly warmer amber" or "make the background pure black").
- **Text rendering:** AI image generators sometimes struggle with exact text. For images 5 (Dry-Run Output) and 6 (Eval Metrics), you may get better results by generating the visual layout from the prompt and then overlaying the text in Figma, Canva, or a similar tool.
- **Consistency pass:** After generating all seven images, review them side by side. If one image drifts in style (e.g., uses gradients when others are flat), regenerate it with an added note: "Match the flat, minimal style of the other images in this series."
- **Export:** Save all final images as PNG at the specified dimensions. Use 72 DPI for web, 144 DPI if you want Retina-ready versions.
