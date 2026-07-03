# Supabase Data Structure Documentation

This document describes the data contract this repository actually depends on: the consolidated `posts` and `profile` tables that `src/data/database.py` reads from Supabase. It is intended to be used as a reference for development and for the Data Science work in this repo that analyzes and predicts post performance.

## 1. Data Flow Overview

**This repo is a consumer, not a producer, of the Supabase data.** It only reads two consolidated, analysis-ready tables — `posts` and `profile` — via `SupabaseClient.get_posts_data()` / `get_profile_data()` in `src/data/database.py`.

Those tables are populated by an **external pipeline that lives outside this repository** (raw ingestion from Notion and social-media APIs, followed by SQL consolidation into `posts`/`profile`). This repo does not track, run, or own that pipeline's code — there is no `notion_supabase_sync.py`, `social_client`, or consolidation SQL checked in here. Section 2.2 below documents the inferred *shape* of the upstream raw tables for context only; treat it as external reference material, not code owned by this project.

```
[External pipeline — not in this repo]
[Notion DBs] --\
                +--> [raw platform/notion tables] --> [SQL consolidation] --> [posts, profile]
[Social APIs] -/                                                                    |
                                                                                      v
                                                                    [This repo reads posts/profile via Supabase]
```

## 2. Table Definitions

The following sections describe the tables in the Supabase database. Section 2.1 (`posts`/`profile`) is the data contract this repo owns and depends on. Section 2.2 is external/upstream reference material — those tables are not read by any code in this repo.

### 2.1. Consolidated Tables

These tables are the primary source for data analysis. They contain aggregated data from various platforms.

#### `posts`

This table consolidates post-related metrics from all social media platforms, with one record per day. The data is pivoted to have separate columns for each platform and for video vs. non-video content.

-   **Primary Key**: `date`

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `date` | `date` | The date of the consolidated record. |
| `post_id_linkedin_no_video` | `text` | The ID of the non-video post on LinkedIn for that day. |
| `posted_at_linkedin_no_video`| `date` | The exact posting date for the non-video LinkedIn post. |
| `num_likes_linkedin_no_video`| `integer` | Number of likes for the non-video LinkedIn post. |
| `num_comments_linkedin_no_video`| `integer` | Number of comments for the non-video LinkedIn post. |
| `num_reshares_linkedin_no_video`| `integer` | Number of reshares for the non-video LinkedIn post. |
| `post_id_instagram_no_video` | `text` | The ID of the non-video post on Instagram. |
| ... | `...` | ...and so on for Instagram, Twitter, Substack, and Threads (non-video). |
| `post_id_linkedin_video` | `text` | The ID of the video post on LinkedIn for that day. |
| `posted_at_linkedin_video` | `date` | The exact posting date for the video LinkedIn post. |
| ... | `...` | ...and so on for all platforms (video). |

*(Note: The table contains a large number of columns. The pattern shown above is repeated for `instagram`, `twitter`, `substack`, and `threads`, for both `_no_video` and `_video` post types.)*

#### `profile`

This table consolidates daily follower counts from all social media platforms.

-   **Primary Key**: `date`

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `date` | `date` | The date of the consolidated record. |
| `num_followers_linkedin` | `integer` | Total number of followers on LinkedIn for that day. |
| `num_followers_instagram` | `integer` | Total number of followers on Instagram for that day. |
| `num_followers_twitter` | `integer` | Total number of followers on Twitter for that day. |
| `num_followers_substack` | `integer` | Total number of followers on Substack for that day. |
| `num_followers_threads` | `integer` | Total number of followers on Threads for that day. |

---

### 2.2. Raw Data Tables *(external reference — not owned by this repo)*

These tables hold the raw, pre-consolidation data. They are populated and consolidated by a pipeline that lives outside this repository; nothing here reads them directly, and their ingestion/consolidation code is not tracked in this repo. They are documented here only so a reader of the consolidated `posts`/`profile` schema understands where the values originate upstream.

#### 2.2.1. Platform-Specific Tables (Inferred)

These tables are the direct inputs for the consolidated tables. Their schema is inferred from the shape of the consolidated data; the SQL that performs the consolidation is part of the external pipeline, not this repo.

-   **`linkedin_posts`** (`post_id`, `posted_at`, `num_likes`, `num_comments`, `num_reshares`, `is_video`)
-   **`instagram_posts`** (`post_id`, `posted_at`, `num_likes`, `num_comments`, `is_video`)
-   **`twitter_posts`** (`post_id`, `posted_at`, `num_likes`, `num_comments`, `num_reshares`, `is_video`)
-   **`substack_posts`** (`post_id`, `posted_at`, `num_likes`, `num_comments`, `num_reshares`, `is_video`)
-   **`threads_posts`** (`post_id`, `posted_at`, `num_likes`, `num_comments`, `num_reshares`, `is_video`)
-   **`linkedin_profile`** (`date`, `num_followers`, `platform`, `data_type`)
-   **`instagram_profile`** (`date`, `num_followers`, `platform`, `data_type`)
-   **`twitter_profile`** (`date`, `num_followers`, `platform`, `data_type`)
-   **`substack_profile`** (`date`, `num_followers`, `platform`, `data_type`)
-   **`threads_profile`** (`date`, `num_followers`, `platform`, `data_type`)

#### 2.2.2. Notion-Synced Tables *(external reference — not owned by this repo)*

These tables are synced from Notion databases by the external ingestion pipeline (selection criteria and sync configuration live in that external system, not in this repo).

**Common Table Structure**

All tables synced from Notion share a common set of columns:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `notion_id` | `text` | The Notion page ID (UUID). **Primary Key**. |
| `created_time` | `timestamp with time zone` | Timestamp of when the page was created in Notion. |
| `last_edited_time`| `timestamp with time zone`| Timestamp of when the page was last edited in Notion. |
| `archived` | `boolean` | Whether the page is archived in Notion. |
| `notion_data_jsonb`| `jsonb` | A JSON blob containing any complex data types (lists, dicts) or properties that could not be mapped to a standard column type. |

**Dynamically Generated Columns**

In addition to the common columns, each table has columns that are dynamically generated based on the properties of the corresponding Notion database. The column names are a "normalized" version of the Notion property name (lowercase, spaces replaced with underscores).

The mapping from Notion property types to PostgreSQL data types is as follows:

| Notion Property Type | PostgreSQL Data Type |
| :--- | :--- |
| Title, Rich Text, URL, Email, Phone | `text` |
| Number | `bigint` or `double precision` |
| Select, Status | `text` |
| Date | `timestamp with time zone` |
| Checkbox | `boolean` |
| Formula (String, Number, Boolean, Date) | `text`, `double precision`, `boolean`, `timestamp with time zone` |
| Multi-Select, Relation, People, Files, Rollup (Array) | Stored in `notion_data_jsonb` |

**List of Replicated Notion Tables**

The following tables are synced from Notion:

-   `notion_editorial`
-   `notion_articles`
-   `notion_books`
-   `notion_books_recommendations`
-   `notion_clips`
-   `notion_comments`
-   `notion_companies`
-   `notion_concepts`
-   `notion_connections`
-   `notion_episodes`
-   `notion_illustrations`
-   `notion_interactions`
-   `notion_newsletter`
-   `notion_posts`
-   `notion_visual_types`
-   `notion_wins_and_features`

**Example: `notion_posts` (Hypothetical Schema)**

Based on a typical "posts" database in Notion, the schema for `notion_posts` might look like this:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `notion_id` | `text` | **Primary Key** |
| `created_time` | `timestamp with time zone` | (Common column) |
| `last_edited_time` | `timestamp with time zone` | (Common column) |
| `archived` | `boolean` | (Common column) |
| `notion_data_jsonb` | `jsonb` | (Common column) |
| `name` | `text` | The title of the post (from a 'Title' property). |
| `status` | `text` | The publishing status (e.g., "Draft", "Published") (from a 'Select' or 'Status' property). |
| `publish_date` | `timestamp with time zone` | The date the post was or is to be published (from a 'Date' property). |
| `platform` | `text` | The target platform (from a 'Select' property). |
| `related_clip` | `jsonb` | Relation to the `notion_clips` table (stored in the JSONB column). |

## 3. Relationships

-   The **consolidated tables** (`posts`, `profile`) are derived from the **platform-specific tables**.
-   The **platform-specific tables** are populated by external API clients and the Notion sync process.
-   Within the **Notion-synced tables**, relationships are not enforced with foreign key constraints at the database level. Instead, they are stored as arrays of Notion page IDs within the `notion_data_jsonb` column (for 'Relation' properties). Any joins between these tables must be performed in the application or query layer by linking `table_a.notion_id` with values from `table_b.notion_data_jsonb`.